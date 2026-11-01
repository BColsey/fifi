//! Single-session CUDA adapter. The shared session owns generation and publication.

use super::{
    cpu_moe::{self, CpuMoeConfig},
    cuda_ffn::device::runtime::{Decoder, LAYER_MAJOR_PIECE_ROWS, State, VOCAB},
    cuda_prefix::{CudaPrefixEnv, CudaPrefixPath, CudaPrefixSnapshot},
    cuda_sample::SamplingScratch,
    exl3_resident::{BankStats, FlintExl3Resident, FlintResidentLimits},
    settings::{
        FlintOptions, LayerMajor, alloc_lazy_return_armed, device_picks_armed, fused_publish_armed,
        fused_publish_suffix, grouped_gather_enabled, kv_device_f16_armed, late_health_selected,
        packed_pointwise_enabled, q8_tiled_armed, resolved_q8_tiled, resolved_switch_suffix,
        token_health_armed,
    },
};
use crate::{
    EngError,
    family::{
        executor::{ExecutorOps, GenerationState},
        project::cuda::{Reserve, served_admission},
        sampler::Sampler,
        session::{
            DecodeSession, Limits, PrefixStore, SaveKind, StoreLimits, TokenStepper, check_prompt,
        },
    },
    kernels::cuda::{
        CudaContext, DeviceStream, plan,
        resident::exl3::{SINGLE_GATHER_VAR, SingleRow, single_gather_from_env},
    },
    sched::job::{PrefixHit, PrefixReport},
    tokenizer::DecodeBatch,
};
use std::path::Path;

type Result<T> = std::result::Result<T, EngError>;

const ALLOC_CACHE_VAR: &str = "FIFI_FLINT_CUDA_ALLOC_CACHE";

/// Layer-major working-memory ceiling: a 21 GiB lease (21,504 MiB).
const LAYER_MAJOR_CEILING_BYTES: u64 = 21 << 30;

/// Rows of one piece of the job-table layer-major pass.
const JOBS_PIECE_ROWS: usize = 1024;

/// The largest block the allocation cache serves in the jobs pass: a tall plane of a piece.
const TALL_BLOCK_BYTES: usize = 64 << 20;

fn alloc_cache_selected() -> bool {
    crate::tokenizer::env_allows_default_on(ALLOC_CACHE_VAR)
}

/// The frozen effective selectors the prefix fingerprint binds: one read per
/// process, shared by every save and restore,
/// so the fingerprint cannot drift when the environment changes mid-run
/// and the three previously-unbound selectors (packed-pointwise, midrows
/// legacy, expert cache) are part of the numeric-path identity.
fn frozen_prefix_env() -> CudaPrefixEnv {
    static ENV: std::sync::OnceLock<CudaPrefixEnv> = std::sync::OnceLock::new();
    ENV.get_or_init(|| CudaPrefixEnv {
        alloc_cache: alloc_cache_selected(),
        late_health: late_health_selected(std::env::var_os("FIFI_FLINT_LATE_HEALTH").as_deref()),
        packed_pointwise: packed_pointwise_enabled(),
        exl3_legacy: std::env::var_os("FIFI_EXL3_LEGACY").as_deref()
            == Some(std::ffi::OsStr::new("1")),
        expert_cache: FlintExl3Resident::cache_forced_on(),
        // Each executor binds its own mode in `prefix_path`.
        layer_major: LayerMajor::Off,
    })
    .clone()
}

#[cfg(test)]
#[path = "cuda_executor_route_trace_tests.rs"]
mod route_trace_tests;
#[cfg(test)]
#[path = "cuda_executor_tests.rs"]
mod tests;
#[cfg(test)]
#[path = "cuda_executor_tf_gate_tests.rs"]
mod tf_gate_tests;

pub(super) struct FlintExecutor {
    // Declared first so the host-plane snapshot store drops BEFORE the
    // device/kernel state (Flash and Q36 declare it first).
    prefix: PrefixStore<CudaPrefixSnapshot, CudaPrefixPath>,
    /// The pinned chat-template digest threaded from the factory: the
    /// park-identity `template` field (identity is assembled at the park
    /// hooks, which own the serializer's row-block size).
    #[allow(dead_code)] // read by the park hooks once they exist
    template_sha: [u8; 32],
    /// The restore fact of the CURRENT job for `prefix_report` (mirror
    /// of the MLX rule): cleared on reset so a job without a
    /// restore never reports the job before it.
    prefix_hit: PrefixHit,
    /// The last restore's uncached-shared observation, taken at the
    /// restore that saw the prompt (no false zeros).
    prefix_uncached_shared: u32,
    /// Lifetime count of saves the EXECUTOR declined before the store
    /// (the zero-committed boundary: a 1-token key): rides
    /// `save_refusals` in the report next to the store's own budget
    /// refusals — the two counters never overlap.
    prefix_zero_committed_saves: u64,
    ctx: CudaContext,
    stream: DeviceStream,
    model: FlintExl3Resident,
    sampling: SamplingScratch,
    eos: Vec<u32>,
    vocab: usize,
    context_limit: usize,
    prefill_chunk_size: usize,
    /// Rows of one layer-major piece when the mode is armed; `None` keeps the
    /// token-major walk and every default untouched.
    layer_major: Option<usize>,
    /// The layer-major pass the engine runs (`Off` keeps the token-major walk).
    layer_mode: LayerMajor,
    bank_tally: BankTally,
    root: std::path::PathBuf,
    options: FlintOptions,
}

/// Expert-bank copies of one request, by phase.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct BankTally {
    prefill_misses: u64,
    decode_misses: u64,
    decode_tokens: u64,
    bytes: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct BankDelta {
    misses: u64,
    bytes: u64,
}

impl BankDelta {
    /// Zero when there is no bank, so a model that holds every expert never counts anything.
    fn between(before: Option<BankStats>, after: Option<BankStats>) -> Self {
        match (before, after) {
            (Some(before), Some(after)) => Self {
                misses: after.misses - before.misses,
                bytes: after.bytes - before.bytes,
            },
            _ => Self::default(),
        }
    }
}

impl BankTally {
    fn begin(&mut self) {
        *self = Self::default();
    }

    fn prefilled(&mut self, delta: BankDelta) {
        self.prefill_misses += delta.misses;
        self.bytes += delta.bytes;
    }

    /// `slots` is the bank's resident count when the request runs on a bank.
    fn decoded(
        &mut self,
        delta: BankDelta,
        tokens: usize,
        finished: bool,
        slots: Option<usize>,
    ) -> Option<String> {
        self.decode_misses += delta.misses;
        self.bytes += delta.bytes;
        self.decode_tokens += tokens as u64;
        let line = slots.filter(|_| finished).map(|slots| self.line(slots))?;
        self.begin();
        Some(line)
    }

    fn line(&self, slots: usize) -> String {
        let per_token = if self.decode_tokens == 0 {
            0.0
        } else {
            self.decode_misses as f64 / self.decode_tokens as f64
        };
        format!(
            "Flint expert bank: slots={slots} prefill_misses={} decode_tokens={} decode_misses={} \
             decode_misses_per_token={per_token:.2} copied_bytes={}",
            self.prefill_misses, self.decode_tokens, self.decode_misses, self.bytes
        )
    }
}

pub(super) struct FlintSession {
    state: State,
    core: DecodeSession,
}

struct Step<'a> {
    executor: &'a FlintExecutor,
    state: &'a mut State,
}

impl Step<'_> {
    fn feed_token_major(&mut self, tokens: &[u32], rows: usize) -> Result<()> {
        if rows == 1 {
            for &token in tokens {
                self.feed(token)?;
            }
        } else {
            let decoder = self.executor.decoder()?;
            for chunk in tokens.chunks(rows) {
                decoder.feed_rows(self.state, chunk)?;
            }
        }
        Ok(())
    }
}

impl TokenStepper for Step<'_> {
    const OPERATION: &'static str = "flint_cuda";

    fn feed(&mut self, token: u32) -> Result<()> {
        self.executor
            .decoder()?
            .step(self.state, token, |_, _| Ok(()))
    }

    fn feed_rows(&mut self, tokens: &[u32]) -> Result<()> {
        let rows = self.executor.prefill_chunk_size;
        let Some(piece_rows) = self.executor.layer_major else {
            return self.feed_token_major(tokens, rows);
        };
        for piece in tokens.chunks(piece_rows) {
            if piece.len() > rows {
                // A jobs pass runs the whole piece as one block; the tile pass keeps narrow ones.
                let block = if self.executor.layer_mode == LayerMajor::Jobs {
                    piece.len()
                } else {
                    rows
                };
                self.executor
                    .decoder()?
                    .feed_rows_layer_major(self.state, piece, block)?;
            } else {
                self.feed_token_major(piece, rows)?;
            }
        }
        Ok(())
    }

    fn logits(&mut self, token: u32) -> Result<Vec<f32>> {
        self.executor
            .decoder()?
            .step(self.state, token, |_, logits| Ok(logits.to_vec()))
    }

    fn next_token(&mut self, token: u32, sampler: &mut Sampler, vocab: usize) -> Result<u32> {
        let executor = self.executor;
        executor.decoder()?.step(self.state, token, |head, logits| {
            executor.sampling.sample(
                &executor.ctx,
                &executor.stream,
                head,
                logits,
                sampler,
                vocab,
            )
        })
    }

    fn fed(&self) -> usize {
        self.state.fed()
    }

    fn reset(&mut self) -> Result<()> {
        self.state.reset(&self.executor.stream)?;
        self.executor.ctx.drain_alloc_cache()?;
        self.executor
            .model
            .recover_expert_cache(&self.executor.ctx, &self.executor.stream)?;
        Ok(())
    }
}

fn work_reserve(plan: &super::kv_plan::KvPlan, chunk_rows: u32, cache_bytes: u64) -> Result<u64> {
    // Six routed + one shared expert retain <16 MiB per extra row at width <=32768.
    ((512 + 16 * (u64::from(chunk_rows) - 1)) << 20)
        .checked_add(State::scratch_bytes(plan)?)
        .and_then(|bytes| bytes.checked_add(cache_bytes))
        .ok_or_else(|| EngError::MemoryPlan("Flint workspace reserve overflows".into()))
}

/// The device memory a load keeps free beside its weights for `options`.
fn reserve_for(options: &FlintOptions, cache_bytes: u64) -> Result<Reserve> {
    reserve_with(options, &options.plan()?, cache_bytes)
}

/// [`reserve_for`] for a KV plan the caller already made, as a jobs pass sizes its rings.
fn reserve_with(
    options: &FlintOptions,
    kv: &super::kv_plan::KvPlan,
    cache_bytes: u64,
) -> Result<Reserve> {
    let (k_row, v_row) = kv.per_plane_row_bytes();
    Ok(Reserve {
        kv_cache: kv.bytes(k_row + v_row)? as u64,
        work: work_reserve(kv, options.prefill_chunk_size, cache_bytes)?,
        headroom: 512 << 20,
    })
}

fn bank_note(context: u32, slots: usize, reason: &str) -> String {
    let why = match reason {
        "auto" => format!("context {context} does not fit with every expert on the GPU"),
        "override" => "FIFI_FLINT_CUDA_EXPERT_SLOTS asks for a smaller bank".to_owned(),
        _ => "FIFI_FLINT_CUDA_EXPERT_CACHE=1 asks for the host buffer".to_owned(),
    };
    format!(
        "Flint expert bank: {why}. {slots} of 256 experts per layer stay on the GPU and all 256 sit in \
         pinned host memory. FIFI_FLINT_CUDA_EXPERT_CACHE=0 keeps every expert on the GPU instead."
    )
}

/// The model's own context limit, which no profile exceeds.
const MODEL_CONTEXT_LIMIT: u32 = 262_144;

/// The largest context for which a load with every expert on the GPU (`resident` weight bytes) is
/// admitted into `free` device bytes, or `None` when not even one token fits. KV cache and work
/// both grow with the context, so the admitted contexts form a prefix.
fn largest_resident_context(
    options: &FlintOptions,
    free: u64,
    resident: u64,
    cache_bytes: u64,
) -> Option<u32> {
    let admits = |context: u32| {
        let options = FlintOptions {
            context_limit: context,
            ..options.clone()
        };
        reserve_for(&options, cache_bytes).is_ok_and(|reserve| {
            let demand = plan::Demand {
                tensors: resident,
                kv_cache: reserve.kv_cache,
                work: reserve.work,
                headroom: reserve.headroom,
            };
            plan::admit(&demand, free).is_ok()
        })
    };
    if !admits(1) {
        return None;
    }
    let (mut fits, mut over) = (1, MODEL_CONTEXT_LIMIT + 1);
    while over - fits > 1 {
        let middle = fits + (over - fits) / 2;
        if admits(middle) {
            fits = middle;
        } else {
            over = middle;
        }
    }
    Some(fits)
}

/// What to tell the operator after a refused load that the host buffer could have avoided, or
/// `None` for a refusal the buffer has nothing to do with.
fn resident_advice(
    forced_off: bool,
    message: &str,
    largest: Option<u32>,
    profile: &str,
) -> Option<String> {
    let host_refused = message.starts_with("Flint expert cache needs");
    if !forced_off && !host_refused {
        return None;
    }
    let fits = largest.map_or_else(
        || String::from("no context fits that way"),
        |tokens| format!("the largest context that fits that way is {tokens} tokens with the {profile} profile, not counting the staging of FIFI_EXL3_LAYER_MAJOR"),
    );
    Some(if forced_off {
        format!(
            "FIFI_FLINT_CUDA_EXPERT_CACHE=0 keeps every expert on the GPU, and {fits}. Unset the \
             variable to keep part of the experts in host memory, or lower context_limit"
        )
    } else {
        format!(
            "Without the host buffer, with every expert on the GPU, {fits}. Lower context_limit \
             to that, or free host memory"
        )
    })
}

fn explain_refusal(
    error: EngError,
    root: &Path,
    options: &FlintOptions,
    free: u64,
    cache_bytes: u64,
) -> EngError {
    let EngError::MemoryPlan(message) = &error else {
        return error;
    };
    let forced_off = FlintExl3Resident::cache_forced_off();
    if !forced_off && !message.starts_with("Flint expert cache needs") {
        return error;
    }
    let largest = FlintExl3Resident::resident_weight_bytes(root)
        .ok()
        .and_then(|resident| largest_resident_context(options, free, resident, cache_bytes));
    match resident_advice(forced_off, message, largest, &options.kv_profile) {
        Some(advice) => EngError::MemoryPlan(format!("{message}. {advice}")),
        None => error,
    }
}

/// `ceiling_bytes` is a parameter, not the const, so the admission tests can
/// exercise the boundary without a device.
fn layer_major_admission(
    switch: &str,
    kv_cache: u64,
    work: u64,
    staging: u64,
    headroom: u64,
    ceiling_bytes: u64,
) -> Result<()> {
    let total = kv_cache
        .checked_add(work)
        .and_then(|bytes| bytes.checked_add(staging))
        .and_then(|bytes| bytes.checked_add(headroom))
        .ok_or_else(|| EngError::MemoryPlan("Flint layer-major demand overflows".into()))?;
    if total > ceiling_bytes {
        // The MiB figures truncate, so a miss of a few bytes would read as
        // equal to the ceiling; the exact overage is stated beside them.
        let mib = |bytes: u64| bytes >> 20;
        return Err(EngError::MemoryPlan(format!(
            "{switch} needs {} MiB (kv {} + work {} + staging {} + headroom {}) \
             over the {} MiB ceiling by {} B; lower context_limit (65K->32K halves the KV term) \
             or unset it",
            mib(total),
            mib(kv_cache),
            mib(work),
            mib(staging),
            mib(headroom),
            mib(ceiling_bytes),
            total - ceiling_bytes
        )));
    }
    Ok(())
}

/// The piece the worker is asked to hand over per `prefill` call. Armed, it is
/// a whole layer-major piece; cancellation is still observed only between
/// pieces, so the armed piece is the cancel granularity.
fn prefill_hint(layer_major: Option<usize>, chunk: usize) -> usize {
    layer_major.unwrap_or(chunk)
}

/// Rows of one layer-major piece: as tall as the router and the pair gathers
/// admit.
fn layer_major_piece_rows(context_rows: usize) -> usize {
    LAYER_MAJOR_PIECE_ROWS.min(context_rows)
}

/// The staging charge of an armed open: what arming allocates for a piece
/// of `piece_rows` — the pair faces for the piece, the three planes and the
/// tile scratch — and nothing more. The pass never stages more than one
/// piece, so a context taller than the piece adds no staging demand (a
/// longer prompt is fed as more pieces); charging faces for the whole
/// context would refuse long contexts for memory the mode never allocates.
fn layer_major_staging_charge(piece_rows: usize, jobs: bool) -> Result<u64> {
    let extra = State::layer_major_extra_bytes(piece_rows, jobs)?;
    let tall = if jobs {
        State::layer_jobs_extra_bytes(piece_rows)?
    } else {
        0
    };
    State::layer_major_staging_bytes(piece_rows)
        .and_then(|faces| faces.checked_add(extra))
        .and_then(|bytes| bytes.checked_add(tall))
        .ok_or_else(|| EngError::MemoryPlan("Flint layer-major staging overflows".into()))
}

/// CPU experts read the pinned host store token by token; neither layer-major pass stages their
/// layers, so the two are refused together before any device work starts. The text names the
/// switch and the options that conflict.
fn cpu_experts_allow(cpu_moe: &CpuMoeConfig, mode: LayerMajor) -> Result<()> {
    if cpu_moe.enabled() && mode != LayerMajor::Off {
        return Err(EngError::BadRequest(
            "FIFI_EXL3_LAYER_MAJOR (1 or jobs) cannot be combined with Flint CPU experts \
             (cpu_moe_offload_layers, cpu_moe_split_experts); unset it or turn the CPU experts off"
                .into(),
        ));
    }
    Ok(())
}

/// The mixed q8k KV profile has no layer-major / jobs prefill arm: both passes
/// stage whole layers, but the q8k write path is only validated on the default
/// token-major launch. Refused at open, before any device work, naming the
/// switch and the profile.
fn q8k_layer_major_allow(profile: &str, mode: LayerMajor) -> Result<()> {
    if profile == "q8k" && mode != LayerMajor::Off {
        return Err(EngError::BadRequest(
            "FIFI_EXL3_LAYER_MAJOR (1 or jobs) has no q8k arm: use kv_profile q8k on the \
             default token-major prefill (unset the switch)"
                .into(),
        ));
    }
    Ok(())
}

/// The strict switches read when the session opens: a bad `FIFI_EXL3_SINGLE_GATHER` or
/// `FIFI_EXL3_LAYER_MAJOR` stops the open, not the first request. Returns the layer-major
/// selection. The single-row switch itself is read once per process by the projection entry; the
/// open only refuses a value that entry would refuse.
fn open_switches(
    single_gather: Option<&std::ffi::OsStr>,
    layer_major: Option<&std::ffi::OsStr>,
) -> Result<LayerMajor> {
    single_gather_from_env(single_gather).map_err(EngError::BadRequest)?;
    super::settings::layer_major_from_env(layer_major)
}

impl FlintExecutor {
    pub(super) fn open(
        root: &Path,
        options: &FlintOptions,
        eos: Vec<u32>,
        vocab: usize,
        template_sha: [u8; 32],
    ) -> Result<(Self, FlintSession)> {
        Self::open_with(root, options, eos, vocab, template_sha, || {
            open_switches(
                std::env::var_os(SINGLE_GATHER_VAR).as_deref(),
                std::env::var_os("FIFI_EXL3_LAYER_MAJOR").as_deref(),
            )
        })
    }

    /// [`Self::open`] with the layer-major selection supplied by the caller,
    /// read at the point `open` has always parsed it, so a test can open
    /// either mode without touching the process environment.
    fn open_with(
        root: &Path,
        options: &FlintOptions,
        eos: Vec<u32>,
        vocab: usize,
        template_sha: [u8; 32],
        layer_major: impl FnOnce() -> Result<LayerMajor>,
    ) -> Result<(Self, FlintSession)> {
        if vocab == 0
            || vocab > VOCAB
            || eos.is_empty()
            || eos.iter().any(|&id| id as usize >= vocab)
        {
            return Err(EngError::Checkpoint(
                "Flint tokenizer vocabulary or EOS set is invalid".into(),
            ));
        }
        // Layer-major prefill: the selection parses strictly now (unset, "1" or "jobs"); the
        // admission check below runs after the reserve is sized.
        let mode = layer_major()?;
        let cpu_moe = CpuMoeConfig::new(
            options.cpu_moe_offload_layers,
            options.cpu_moe_split_experts,
            options.cpu_moe_threads,
        )?;
        cpu_experts_allow(&cpu_moe, mode)?;
        q8k_layer_major_allow(&options.kv_profile, mode)?;
        let context_rows = options.context_limit as usize;
        let piece_rows = match mode {
            LayerMajor::Off => None,
            LayerMajor::Tiles => Some(layer_major_piece_rows(context_rows)),
            LayerMajor::Jobs => Some(JOBS_PIECE_ROWS.min(context_rows)),
        };
        // A jobs pass appends a whole piece to the rings at once, so they hold a piece.
        let kv = match (mode, piece_rows) {
            (LayerMajor::Jobs, Some(rows)) => options.plan_rows(rows)?,
            _ => options.plan()?,
        };
        let cache_enabled = alloc_cache_selected();
        let ctx = CudaContext::with_admission_cache(&served_admission(), cache_enabled)?;
        // A Flint session runs its single-row projections on the wide kernel unless the operator
        // names another in FIFI_EXL3_SINGLE_GATHER (`off` is the shuffle kernel); same bytes, faster.
        ctx.set_single_row_default(SingleRow::Wide(1));
        let stream = ctx.stream()?;
        let lazy_return = cache_enabled && alloc_lazy_return_armed();
        if let (LayerMajor::Jobs, Some(rows), true) = (mode, piece_rows, cache_enabled) {
            // Tall frames are megabytes: let the cache hold the working set of one layer. The
            // limits belong to this session's context for as long as it lives; the jobs pass is
            // chosen at open, so nothing sets them back.
            ctx.set_alloc_cache_limits(
                TALL_BLOCK_BYTES,
                usize::try_from(State::layer_jobs_idle_bytes(rows)?).unwrap_or(TALL_BLOCK_BYTES),
            );
        }
        ctx.set_lazy_alloc_return(lazy_return);
        let free = ctx.memory_info()?.free as u64;
        let cache_bytes = CudaContext::alloc_cache_reserve(cache_enabled);
        let mut reserve = reserve_with(options, &kv, cache_bytes)?;
        let layer_major_rows = if let Some(rows) = piece_rows {
            let staging = layer_major_staging_charge(rows, mode == LayerMajor::Jobs)?;
            // The mode's staging is part of its declared demand: admission
            // refuses above the 21,504 MiB working ceiling with the figure
            // and the KV-cap remedy, before any weight is loaded. The charge
            // is what arming allocates, so the weights leave exactly the room
            // the pass needs.
            layer_major_admission(
                if mode == LayerMajor::Jobs {
                    "FIFI_EXL3_LAYER_MAJOR=jobs"
                } else {
                    "FIFI_EXL3_LAYER_MAJOR=1"
                },
                reserve.kv_cache,
                reserve.work,
                staging,
                reserve.headroom,
                LAYER_MAJOR_CEILING_BYTES,
            )?;
            reserve.work = reserve.work.checked_add(staging).ok_or_else(|| {
                EngError::MemoryPlan("Flint layer-major work reserve overflows".into())
            })?;
            Some(rows)
        } else {
            None
        };
        cpu_moe::install(cpu_moe)?;
        let model = FlintExl3Resident::load_with_prefill(
            &ctx,
            &stream,
            root,
            FlintResidentLimits {
                max_staging_bytes: 2 << 30,
                max_weight_bytes: free,
                reserve,
            },
            options.prefill_chunk_size as usize,
        )
        .map_err(|error| explain_refusal(error, root, options, free, cache_bytes))?;
        plan::admit(
            &plan::Demand {
                tensors: 0,
                kv_cache: reserve.kv_cache,
                work: reserve.work,
                headroom: reserve.headroom,
            },
            ctx.memory_info()?.free as u64,
        )?;
        Decoder::new(&ctx, &stream, &model)?;
        let mut state = State::new(&ctx, kv)?;
        if let Some(rows) = layer_major_rows {
            state.arm_layer_major(&ctx, rows, mode == LayerMajor::Jobs)?;
        }
        let sampling = SamplingScratch::new(&ctx)?;
        eprintln!(
            "Flint CUDA V/{}-write {}: context={} kv={} kv_bytes={} work_reserve={} headroom={} weights={} alloc_cache={} late_health={} token_health={}{}{}{}{}{}{}",
            options.kv_profile.to_ascii_uppercase(),
            if options.kv_profile == "q8" || options.kv_profile == "q8k" {
                "native"
            } else {
                "bridge"
            },
            options.context_limit,
            options.kv_profile,
            reserve.kv_cache,
            reserve.work,
            reserve.headroom,
            model.device_weight_bytes(),
            if cache_enabled { "on" } else { "off" },
            if late_health_selected(std::env::var_os("FIFI_FLINT_LATE_HEALTH").as_deref()) {
                "on"
            } else {
                "off"
            },
            if token_health_armed() { "on" } else { "off" },
            resolved_switch_suffix(
                packed_pointwise_enabled(),
                grouped_gather_enabled(),
                resolved_q8_tiled(&options.kv_profile, q8_tiled_armed),
                kv_device_f16_armed(),
            ),
            fused_publish_suffix(fused_publish_armed()),
            // Printed only when armed, so the default load line is unchanged.
            layer_major_rows.map_or(String::new(), |rows| {
                let jobs = if mode == LayerMajor::Jobs {
                    " layer_jobs=on"
                } else {
                    ""
                };
                format!(" layer_major={rows}{jobs}")
            }),
            ctx.single_gather_suffix(),
            model
                .banked_experts()
                .zip(model.bank_reason())
                .map_or(String::new(), |(slots, reason)| {
                    format!(" expert_slots={slots} bank={reason}")
                }),
            if lazy_return {
                " alloc_return=lazy"
            } else {
                ""
            },
        );
        if let Some((slots, reason)) = model.banked_experts().zip(model.bank_reason()) {
            eprintln!("{}", bank_note(options.context_limit, slots, reason));
        }
        if let Some(moe) = cpu_moe::active() {
            eprintln!("{}", moe.describe());
        }
        if device_picks_armed() {
            eprintln!("Flint device picks: on (single-row decode steps over a resident load)");
        }
        // Resolve the prefix opt-in against the kv profile at startup.
        // This fires the one-time load-line note when the env is set but
        // the profile refuses, instead of failing the first boundary
        // request.
        let _ = super::prefix_reuse_enabled_for(&options.kv_profile);
        Ok((
            Self {
                ctx,
                stream,
                model,
                sampling,
                eos,
                vocab,
                context_limit: options.context_limit as usize,
                prefill_chunk_size: options.prefill_chunk_size as usize,
                layer_major: layer_major_rows,
                layer_mode: mode,
                bank_tally: BankTally::default(),
                root: root.to_path_buf(),
                template_sha,
                // The one-entry tip store (the MLX mirror); serving default-off is
                // the family gate in run.rs (flag + FIFI_FLINT_PREFIX_REUSE).
                prefix: PrefixStore::with_limits(StoreLimits::default()),
                prefix_hit: PrefixHit::Miss,
                prefix_uncached_shared: 0,
                prefix_zero_committed_saves: 0,
                options: options.clone(),
            },
            FlintSession {
                state,
                core: DecodeSession::default(),
            },
        ))
    }

    /// The store fingerprint computed from the CURRENT frozen selectors —
    /// the effective values the numeric path reads, never raw env.
    /// All five selectors freeze together, once per process, so a save and
    /// a later restore can never see different spellings of one state.
    fn prefix_path(&self) -> CudaPrefixPath {
        // The mode is the executor's own, fixed at open: layer-major KV is
        // computed by different matrix kernels, so a prefix saved under one
        // mode must not restore under the other.
        let env = CudaPrefixEnv {
            layer_major: self.layer_mode,
            ..frozen_prefix_env()
        };
        CudaPrefixPath::now(&self.root, &self.options, env)
    }

    /// Default OFF stays bit-for-bit unobservable.
    fn prefix_armed(&self) -> bool {
        super::prefix_reuse_enabled_for(&self.options.kv_profile)
    }

    fn decoder(&self) -> Result<Decoder<'_>> {
        Decoder::new(&self.ctx, &self.stream, &self.model)
    }

    /// One logits row without the sampler, for the qualification greedy
    /// walk and the bitwise boundary-row comparisons.
    #[cfg(test)]
    pub(super) fn qual_logits(&self, session: &mut FlintSession, token: u32) -> Result<Vec<f32>> {
        Step {
            executor: self,
            state: &mut session.state,
        }
        .logits(token)
    }
}

impl FlintSession {
    /// The boundary token a fed prompt left pending. Visible to the prefix
    /// tests because `core` is private to this module.
    #[cfg(test)]
    pub(super) fn pending_token(&self) -> Option<u32> {
        self.core.pending()
    }

    /// Rows the ring has fed. The wrap gate asserts this reaches the tip so
    /// a silently truncated wrap arm cannot degrade into a fresh comparison.
    #[cfg(test)]
    pub(super) fn fed(&self) -> usize {
        self.state.fed()
    }
}

impl ExecutorOps for FlintExecutor {
    type Session = FlintSession;

    fn check_prompt(&self, tokens: usize) -> Result<()> {
        check_prompt(tokens, self.context_limit)
    }

    fn history<'s>(&self, session: &'s FlintSession) -> &'s [u32] {
        session.core.history()
    }

    fn reset(&mut self, session: &mut FlintSession) -> Result<()> {
        // A job without a restore never reports the job before it:
        // the restore facts clear with the session.
        self.prefix_hit = PrefixHit::Miss;
        self.prefix_uncached_shared = 0;
        self.bank_tally.begin();
        session.core.reset(&mut Step {
            executor: self,
            state: &mut session.state,
        })
    }

    fn prefill(&mut self, session: &mut FlintSession, tokens: &[u32]) -> Result<()> {
        let before = self.model.bank_stats();
        let result = session.core.prefill(
            &mut Step {
                executor: self,
                state: &mut session.state,
            },
            tokens,
            self.context_limit,
        );
        let delta = BankDelta::between(before, self.model.bank_stats());
        self.bank_tally.prefilled(delta);
        result
    }

    fn decode(
        &mut self,
        session: &mut FlintSession,
        generation: &mut GenerationState,
    ) -> Result<DecodeBatch> {
        let limits = Limits {
            eos: &self.eos,
            sampler_vocab: self.vocab,
            context_limit: self.context_limit,
        };
        let before = self.model.bank_stats();
        let batch = session.core.decode(
            &mut Step {
                executor: self,
                state: &mut session.state,
            },
            generation,
            &limits,
        )?;
        let delta = BankDelta::between(before, self.model.bank_stats());
        let finished = batch.finish.is_some();
        if finished && let Some(moe) = cpu_moe::active() {
            eprintln!("{}", moe.summary());
        }
        let slots = self.model.banked_experts();
        if let Some(line) = self
            .bank_tally
            .decoded(delta, batch.tokens.len(), finished, slots)
        {
            eprintln!("{line}");
        }
        Ok(batch)
    }

    fn boundary(&mut self, _session: &mut FlintSession) {}

    fn prefill_piece(&self) -> Option<usize> {
        Some(prefill_hint(self.layer_major, self.prefill_chunk_size))
    }

    /// Saves the session's committed KV planes as the store's tip entry
    /// (slice P1). An unarmed engine (opt-in off, or resolved off by a
    /// non-f16 profile) reports "not saved" WITHOUT error before the
    /// profile gate, so a q8 boundary request serves cold;
    /// the refusal is a planned no-op, never a BadRequest.
    fn save_prefix(&mut self, st: &FlintSession, key: &[u32], kind: SaveKind) -> Result<bool> {
        if !self.prefix_armed() {
            return Ok(false);
        }
        super::cuda_prefix::profile_allows_prefix(&self.options.kv_profile)?;
        // A zero-committed save: a 1-token key
        // leaves nothing committed, and the committed-boundary check
        // would fail the REQUEST. Decline to cache instead — the boundary
        // request serves cold with no error — and count the decline
        // executor-side so `save_refusals` shows it.
        if st.state.fed() == 0 {
            self.prefix_zero_committed_saves = self.prefix_zero_committed_saves.saturating_add(1);
            return Ok(false);
        }
        let path = self.prefix_path();
        let fed = st.state.fed();
        let ctx = &self.ctx;
        let stream = &self.stream;
        st.core
            .save_prefix(&mut self.prefix, key, path, fed, kind, || {
                st.state.snapshot_prefix(ctx, stream)
            })
    }

    /// Replaces the reset session's device state with the saved tip when the
    /// prompt extends it and the numeric path matches. Unarmed engines
    /// reuse 0 without error, same rule as the save.
    fn restore_prefix(&mut self, st: &mut FlintSession, prompt: &[u32]) -> Result<usize> {
        if !self.prefix_armed() {
            return Ok(0);
        }
        super::cuda_prefix::profile_allows_prefix(&self.options.kv_profile)?;
        let path = self.prefix_path();
        let core = &mut st.core;
        let state = &mut st.state;
        let ctx = &self.ctx;
        let stream = &self.stream;
        let restored = core.restore_prefix(&mut self.prefix, prompt, path, |snapshot| {
            state.restore_prefix(ctx, stream, snapshot)
        })?;
        self.prefix_hit = if restored > 0 {
            match self.prefix.restored_kind() {
                Some(SaveKind::Root) => PrefixHit::Root,
                Some(SaveKind::Periodic) => PrefixHit::Periodic,
                Some(SaveKind::Tip) | Some(SaveKind::Block) | None => PrefixHit::Tip,
            }
        } else {
            PrefixHit::Miss
        };
        self.prefix_uncached_shared = self.prefix.uncached_shared(prompt, restored);
        Ok(restored)
    }

    fn prefix_limits(&self) -> StoreLimits {
        if self.prefix_armed() {
            StoreLimits::default()
        } else {
            StoreLimits {
                roots: 0,
                entries: 0,
            }
        }
    }

    fn prefix_report(&self) -> Option<PrefixReport> {
        if !self.prefix_armed() {
            return None;
        }
        let stored = self.prefix.roots_len() + self.prefix.entries_len();
        let hit = self.prefix_hit;
        let mut save_side = self.prefix.save_side_totals();
        save_side.save_refusals = save_side
            .save_refusals
            .saturating_add(self.prefix_zero_committed_saves);
        Some(PrefixReport {
            entries: u32::try_from(stored).unwrap_or(u32::MAX),
            bytes: self.prefix.snapshots().map(|s| s.host_bytes()).sum(),
            hit,
            uncached_shared: self.prefix_uncached_shared,
            // The shadow stays off for CUDA: the block-match flag is not
            // lifted for this store yet.
            shadow: None,
            save_side,
            restore_drops: self.prefix.restore_drop_count(),
            evictions: self.prefix.eviction_count(),
        })
    }
}

#[cfg(test)]
impl FlintExecutor {
    pub(super) fn open_qualification(limit: u32, profile: &str) -> (Self, FlintSession) {
        let root = std::env::var_os("FLINT_QUAL_MODEL")
            .map_or_else(|| Path::new("/not-set").to_path_buf(), Into::into);
        let tokenizer_dir =
            std::env::var_os("FLINT_QUAL_TOKENIZER").map_or_else(|| root.clone(), Into::into);
        let options = FlintOptions {
            context_limit: limit,
            kv_profile: profile.to_owned(),
            prefill_chunk_size: 8,

            cpu_moe_offload_layers: 0,
            cpu_moe_split_experts: 0,
            cpu_moe_threads: 0,
        };
        let bundle =
            super::factory::load_tokenizer(&tokenizer_dir).expect("the qualification tokenizer");
        let eos = bundle.eos_token_ids().to_vec();
        let vocab = bundle.vocab_size() as usize;
        Self::open(
            &root,
            &options,
            eos,
            vocab,
            super::factory::pinned_template_sha(),
        )
        .expect("the qualification executor")
    }

    /// Envelope-gate probe: (free, total) device bytes for the 40 GiB
    /// guard assertion and the envelope record.
    pub(super) fn qual_memory_info(&self) -> (u64, u64) {
        let info = self.ctx.memory_info().expect("device memory info");
        (info.free as u64, info.total as u64)
    }
}

#[cfg(test)]
mod bank_tally_tests {
    use super::*;

    #[test]
    fn tally_reports_misses_per_decode_token_and_copied_bytes() {
        let tally = BankTally {
            prefill_misses: 31_066,
            decode_misses: 1_108,
            decode_tokens: 128,
            bytes: 32_174 * 1_603_584,
        };
        assert_eq!(
            tally.line(224),
            format!(
                "Flint expert bank: slots=224 prefill_misses=31066 decode_tokens=128 \
                 decode_misses=1108 decode_misses_per_token=8.66 copied_bytes={}",
                32_174u64 * 1_603_584
            )
        );
        assert!(
            BankTally::default()
                .line(48)
                .contains("decode_misses_per_token=0.00")
        );
    }

    #[test]
    fn misses_are_the_growth_of_the_bank_counters_and_zero_without_a_bank() {
        let at = |misses, bytes| Some(BankStats { misses, bytes });
        let grew = |misses, bytes| BankDelta { misses, bytes };
        assert_eq!(BankDelta::between(at(5, 80), at(9, 144)), grew(4, 64));
        assert_eq!(BankDelta::between(at(5, 80), at(5, 80)), grew(0, 0));
        assert_eq!(BankDelta::between(None, None), grew(0, 0));
        assert_eq!(BankDelta::between(None, at(9, 144)), grew(0, 0));
    }

    fn delta(misses: u64, bytes: u64) -> BankDelta {
        BankDelta { misses, bytes }
    }

    #[test]
    fn a_request_reports_its_own_counts_once_and_the_next_starts_from_zero() {
        let mut tally = BankTally::default();
        tally.prefilled(delta(100, 1_000));
        tally.prefilled(delta(20, 200));
        assert_eq!(tally.decoded(delta(5, 50), 4, false, Some(224)), None);
        let line = tally.decoded(delta(3, 30), 2, true, Some(224)).unwrap();
        assert_eq!(
            line,
            "Flint expert bank: slots=224 prefill_misses=120 decode_tokens=6 decode_misses=8 \
             decode_misses_per_token=1.33 copied_bytes=1280"
        );
        let line = tally.decoded(delta(1, 10), 1, true, Some(224)).unwrap();
        assert!(
            line.contains("prefill_misses=0 decode_tokens=1 decode_misses=1"),
            "{line}"
        );
        assert!(line.ends_with("copied_bytes=10"), "{line}");
    }

    #[test]
    fn a_new_request_discards_an_unfinished_one_and_a_full_bank_never_reports() {
        let mut tally = BankTally::default();
        tally.prefilled(delta(7, 70));
        assert_eq!(tally.decoded(delta(2, 20), 3, false, Some(48)), None);
        tally.begin();
        assert_eq!(tally, BankTally::default());
        assert_eq!(tally.decoded(BankDelta::default(), 5, true, None), None);
        assert_eq!(tally.decode_tokens, 5);
        tally.begin();
        let line = tally.decoded(delta(1, 10), 4, true, Some(48)).unwrap();
        assert!(
            line.contains("slots=48 prefill_misses=0 decode_tokens=4 decode_misses=1"),
            "{line}"
        );
    }
}

#[cfg(test)]
mod resident_advice_tests {
    use super::*;

    // The weights of the published pack with every expert resident, from the load line.
    const RESIDENT: u64 = 21_469_896_448;

    fn options(context_limit: u32, kv_profile: &str) -> FlintOptions {
        FlintOptions {
            context_limit,
            kv_profile: kv_profile.to_owned(),
            prefill_chunk_size: 8,

            cpu_moe_offload_layers: 0,
            cpu_moe_split_experts: 0,
            cpu_moe_threads: 0,
        }
    }

    fn admits(options: &FlintOptions, free: u64, cache_bytes: u64) -> bool {
        let reserve = reserve_for(options, cache_bytes).unwrap();
        let demand = plan::Demand {
            tensors: RESIDENT,
            kv_cache: reserve.kv_cache,
            work: reserve.work,
            headroom: reserve.headroom,
        };
        plan::admit(&demand, free).is_ok()
    }

    #[test]
    fn the_largest_resident_context_admits_and_one_more_token_does_not() {
        let cache_bytes = CudaContext::alloc_cache_reserve(true);
        for (profile, free) in [
            ("f16", 24_000_000_000u64),
            ("f16", 23_500_000_000),
            ("q8", 24_000_000_000),
        ] {
            let base = options(1, profile);
            let largest = largest_resident_context(&base, free, RESIDENT, cache_bytes).unwrap();
            assert!(
                admits(&options(largest, profile), free, cache_bytes),
                "{profile} {largest}"
            );
            assert!(
                largest == MODEL_CONTEXT_LIMIT
                    || !admits(&options(largest + 1, profile), free, cache_bytes),
                "{profile} {largest}"
            );
        }
        // The published ceilings are about 45K tokens for f16 and 86K for q8 on this card.
        let f16 =
            largest_resident_context(&options(1, "f16"), 24_000_000_000, RESIDENT, cache_bytes);
        assert!((40_000..50_000).contains(&f16.unwrap()), "{f16:?}");
        assert_eq!(
            largest_resident_context(&options(1, "f16"), RESIDENT, RESIDENT, cache_bytes),
            None
        );
    }

    #[test]
    fn the_startup_note_says_why_the_bank_is_on() {
        let auto = bank_note(131_072, 224, "auto");
        assert!(
            auto.contains("context 131072 does not fit with every expert on the GPU"),
            "{auto}"
        );
        assert!(auto.contains("224 of 256 experts per layer"), "{auto}");
        assert!(auto.contains("FIFI_FLINT_CUDA_EXPERT_CACHE=0"), "{auto}");
        assert!(bank_note(4096, 256 - 8, "forced").contains("CACHE=1 asks for the host buffer"));
        assert!(bank_note(4096, 192, "override").contains("EXPERT_SLOTS asks for a smaller bank"));
    }

    #[test]
    fn a_refused_resident_load_names_the_switch_and_the_context_that_fits() {
        let host = "Flint expert cache needs 20 bytes plus 8 bytes for the OS floor. Available host/cgroup budget: 9 bytes";
        let off =
            resident_advice(true, "cuda admission: need 1 bytes", Some(45_315), "f16").unwrap();
        assert!(off.contains("FIFI_FLINT_CUDA_EXPERT_CACHE=0"), "{off}");
        assert!(off.contains("45315 tokens") && off.contains("f16"), "{off}");
        assert!(
            off.contains("Unset the variable") && off.contains("lower context_limit"),
            "{off}"
        );
        let none = resident_advice(true, "cuda admission: need 1 bytes", None, "q8").unwrap();
        assert!(none.contains("no context fits that way"), "{none}");
        let refused = resident_advice(false, host, Some(45_315), "f16").unwrap();
        assert!(
            refused.contains("Without the host buffer") && refused.contains("45315 tokens"),
            "{refused}"
        );
        assert!(
            !refused.contains("FIFI_FLINT_CUDA_EXPERT_CACHE=0"),
            "{refused}"
        );
        assert_eq!(
            resident_advice(false, "cuda admission: need 1 bytes", Some(7), "f16"),
            None
        );
    }
}

#[cfg(test)]
mod open_switch_tests {
    use super::{LayerMajor, open_switches};
    use std::ffi::OsStr;

    #[test]
    fn a_bad_single_row_switch_stops_the_open() {
        let open = |single: Option<&str>, major: Option<&str>| {
            open_switches(single.map(OsStr::new), major.map(OsStr::new))
        };
        assert_eq!(open(None, None).unwrap(), LayerMajor::Off);
        assert_eq!(open(Some("1"), None).unwrap(), LayerMajor::Off);
        assert_eq!(open(Some("1"), Some("1")).unwrap(), LayerMajor::Tiles);
        assert_eq!(open(None, Some("jobs")).unwrap(), LayerMajor::Jobs);
        assert!(open(Some("wide2"), None).is_ok());
        assert!(open(Some("wide8"), None).is_err());
        for bad in ["", "0", "true", "on", "gather"] {
            let error = open(Some(bad), None).expect_err("a bad value stops the open");
            let text = format!("{error:?}");
            assert!(text.contains("FIFI_EXL3_SINGLE_GATHER"), "{bad}: {text}");
        }
        assert!(open(Some("1"), Some("2")).is_err());
        assert!(open(None, Some("")).is_err());
    }

    #[test]
    fn the_open_reads_both_switches_through_open_switches() {
        let source = include_str!("cuda_executor.rs");
        let open = source.find("pub(super) fn open(").expect("open");
        let body = &source[open..open + 600];
        assert!(body.contains("open_switches("), "{body}");
        assert!(body.contains("SINGLE_GATHER_VAR"), "{body}");
        assert!(
            !body.contains("layer_major_from_env("),
            "the open reads the switches only through open_switches"
        );
    }

    #[test]
    fn q8k_refuses_both_layer_major_arms_at_open() {
        for mode in [LayerMajor::Tiles, LayerMajor::Jobs] {
            let error =
                super::q8k_layer_major_allow("q8k", mode).expect_err("q8k has no layer-major arm");
            let text = format!("{error:?}");
            assert!(text.contains("q8k"), "{text}");
            assert!(text.contains("FIFI_EXL3_LAYER_MAJOR"), "{text}");
            assert!(text.contains("token-major"), "{text}");
        }
        assert!(super::q8k_layer_major_allow("q8k", LayerMajor::Off).is_ok());
        assert!(super::q8k_layer_major_allow("f16", LayerMajor::Jobs).is_ok());
        assert!(super::q8k_layer_major_allow("q8", LayerMajor::Tiles).is_ok());
    }
}
