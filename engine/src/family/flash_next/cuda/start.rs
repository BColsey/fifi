//! Starts the Flash-Next CUDA engine of one `enginesrv` config entry: the tokenizer bundle and the
//! rules on the calling thread, then the engine thread builds everything CUDA inside `make` (no
//! CUDA value crosses a thread boundary). It follows `qwen36/cuda/start.rs`.
//!
//! Load order inside `make`, each stage announced to the engine state:
//! 1. the CUDA context, and the card's free memory before any weight is resident;
//! 2. the pack, opened through one local-storage guard (`FlashPackFiles`);
//! 3. the memory plan for the configured context (`FlashLoadPlan::at_context`): the expert slots
//!    per layer that fit next to the trunk and the K/V, or a refusal that names the longest context;
//! 4. the trunk, uploaded in the order of the walk (`trunk_weights`);
//! 5. the expert bank: device slots per layer filled with the first experts of each layer, and the
//!    rest pinned in host RAM (`DeviceBank::build`), after the host plan and the cgroup check;
//! 6. the decode step and the split MoE block over the bank;
//! 7. the host feeds: the embedding table and the PLE block (host projections), the n-gram table.
//!
//! Lifetime: `spawn_engine` needs an executor without a lifetime and the step borrows its owners, so
//! the projector, the stream, the checkpoint and the weights are `Box::leak`ed once per engine.
//! ONE CUDA engine is built per process (the same rule as the Qwen3.6 CUDA engine): a guard pins
//! the flag before the first leak, and a retry in the same process is refused.

use std::{
    cell::Cell,
    path::Path,
    rc::Rc,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

use super::{
    bank::{HostMemory, HostPinPlan, HostTerms, RecordSource, device::DeviceBank},
    bank_source::{self, PackRecordSource},
    device::{CardChunkDevice, CardDevice},
    driver::chunk::ChunkDriver,
    driver::{
        DriverShape, ResumeSnapshot, ResumesState, StepperPositions, TokenDriver, TokenTimes,
    },
    executor::{CudaRules, FlashCudaExecutor},
    feeds::{
        DEFAULT_THREADS, StartShape,
        device::{DevicePle, PleDeviceShape},
    },
    load::{TrunkList, upload::trunk_weights},
    moe_block::{SplitMoe, SplitMoeTrunk},
    pack_feeds::PackFeeds,
    ple_feed::{FixedProjection, NgramRows, PleFeed, PleVectors},
    prefill_adapter::PrefillWaveScope,
    state::FlashGeometry,
    step::{FlashStep, SplitMoeAt},
    step_plan::StepPlan,
};
use crate::family::engine_announce::{Announcement, announce_failure, await_announced};
use crate::family::expert_bank::{
    BankGeometry, ExpertBankState, HostMode, ReplaceOnMissLru,
    bank_tier::{RUNG_RING_SLOTS, TierChoice, rung_ring_placeholder_bytes},
};
use crate::family::flash_next::settings::{ngram_bulk, ple_batch, ple_device};
use crate::{
    EngError,
    family::{
        device_budget::KvPlan,
        exl3::source::PackReader,
        exl3_pack::layer_name,
        flash_next::{
            artifact::{pack_identity, tokenizer_pin},
            load::{
                model_shape,
                pack_files::{FlashPackFiles, NGRAM_FILE},
            },
            moe::{
                load_plan::FlashLoadPlan,
                tier_plan::{TierInputs, TierShape},
            },
            settings::{CudaSettings, FlashSettings},
        },
        mem::Facts,
        project::cuda::{CudaProjector, CudaWeight, CudaWeights, Reserve, exl3::Exl3Weight},
        session::DecodeSession,
        session::PrefixIdentity,
    },
    kernels::cuda::{
        CudaContext, DeviceBuffer, DeviceStream, host_register,
        resident::{
            exl3::{IndexedOptions, SERVED_SINGLE_ROW, ServedKernels},
            hyper::HyperDtype,
        },
    },
    sched::worker::{Engine, ReadyNote, Stages, spawn_engine_noted},
    tensor::Dtype,
    tokenizer::TokenizerBundle,
    weights::exl3::Checkpoint,
};

/// Threads of the host PLE projections (rows split over them). The two projection matvecs are
/// DRAM-bandwidth-bound (131 MB per token), so more threads helps until the socket saturates;
/// `INFERENCE_FLASH_NEXT_PLE_THREADS` overrides the default.
fn ple_threads() -> usize {
    std::env::var("INFERENCE_FLASH_NEXT_PLE_THREADS")
        .ok()
        .and_then(|text| text.parse::<usize>().ok())
        .map(|n| n.clamp(1, 128))
        .unwrap_or(8)
}

/// The id-indexed options Flash-Next serves its routed experts with: the DP4A decode of codebook 2
/// (the codebook of this pack), one launch for the 20 gate and up pairs of ten picks, and the
/// pipelined tile loop on: it writes the bytes of the default launches and the serve A/B
/// (alternating arms, two rounds, server metrics) took 8-11% off the time to first token at
/// every depth and added 5-8% decode tokens per second. A set `FIFI_EXL3_DP4A`,
/// `FIFI_EXL3_INDEXED_PAIRS` or `FIFI_EXL3_PIPELINE` still wins for its own option.
const SERVED_INDEXED: IndexedOptions = IndexedOptions {
    dp4a: true,
    pairs20: true,
    pipeline: true,
    table: false,
};

/// Registers the kernels the Flash-Next decode runs on its context: the wide single-row kernel for
/// every dense projection and the id-indexed options of the routed experts. Qwen3.6 and Flint
/// register their own; a bad launch switch or an option the kernel cannot combine with is refused
/// here, before any weight is uploaded.
fn register_served_kernels(ctx: &CudaContext) -> Result<(), EngError> {
    ctx.register_served_kernels(ServedKernels {
        single_row: SERVED_SINGLE_ROW,
        indexed: Some(SERVED_INDEXED),
        odd_rates: true,
    })?;
    eprintln!(
        "flash-next cuda: single-row kernel {}{}",
        ctx.single_row_kernel_name()?,
        ctx.single_gather_suffix()
    );
    Ok(())
}

static CUDA_ENGINE_BUILT: AtomicBool = AtomicBool::new(false);

fn already_built() -> EngError {
    EngError::BadRequest(
        "engine config: the Flash-Next CUDA backend builds one engine per process: its weights are leaked for the process lifetime; configure a single qwen3_8_flash_next entry and restart the process to reload"
            .to_string(),
    )
}

/// Takes the one-engine flag; released on drop unless committed (a failure before the first leak
/// owns nothing on the card).
pub(super) struct BuildGuard {
    committed: bool,
}

impl BuildGuard {
    fn acquire() -> Result<Self, EngError> {
        CUDA_ENGINE_BUILT
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| already_built())?;
        Ok(Self { committed: false })
    }

    fn commit(&mut self) {
        self.committed = true;
    }
}

impl Drop for BuildGuard {
    fn drop(&mut self) {
        if !self.committed {
            CUDA_ENGINE_BUILT.store(false, Ordering::Release);
        }
    }
}

/// Loads and pins the tokenizer, then starts the engine thread, which loads the pack. The engine
/// state stays `Loading` until the executor exists.
pub(in crate::family::flash_next) fn start(
    settings: &FlashSettings,
    ready_note: ReadyNote,
) -> Result<(Engine, TokenizerBundle, u32), EngError> {
    let cuda = settings.cuda.as_ref().ok_or_else(|| {
        EngError::BadRequest("engine config: the CUDA decoder is not enabled".to_string())
    })?;
    let manifest = std::fs::read(&settings.manifest).map_err(|error| {
        EngError::Checkpoint(format!(
            "the manifest {} cannot be read: {error}",
            settings.manifest.display()
        ))
    })?;
    let pin = tokenizer_pin(&manifest, settings.tokenizer_vocab_size)?;
    let identity = pack_identity(&settings.artifact, &pin)?;
    let bundle = TokenizerBundle::load(&settings.tokenizer_dir, &pin)?;
    let rules = CudaRules {
        eos: Arc::from(bundle.eos_token_ids()),
        sampler_vocab: usize::try_from(bundle.vocab_size())
            .map_err(|_| EngError::Checkpoint("tokenizer vocabulary exceeds usize".to_string()))?,
    };
    let prefill_chunk = usize::try_from(settings.prefill_chunk)
        .map_err(|_| EngError::BadRequest("engine config: prefill_chunk exceeds usize".into()))?;
    let artifact = settings.artifact.clone();
    let wanted = cuda.clone();
    let guard = BuildGuard::acquire()?;
    // The engine thread plans the memory and sends the context it chose, once, when the config
    // sets none; the server's request bound needs it before the weights finish loading.
    let (announce, resolved) = if settings.context_auto {
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        (Some(sender), Some(receiver))
    } else {
        (None, None)
    };
    let engine = spawn_engine_noted(prefill_chunk, ready_note, move |stages| {
        make(
            stages,
            &artifact,
            &wanted,
            announce,
            rules,
            identity,
            prefill_chunk as u32,
            guard,
        )
    })?;
    let context = match resolved {
        Some(receiver) => await_announced(&receiver, "Flash-Next", || engine.handle().failure())?,
        None => cuda.context,
    };
    Ok((engine, bundle, context))
}

fn raw<'w>(weights: &'w CudaWeights, name: &str) -> Result<&'w DeviceBuffer, EngError> {
    match weights.get(name) {
        Some(CudaWeight::Raw { buffer, .. }) => Ok(buffer),
        _ => Err(EngError::Checkpoint(format!(
            "Flash-Next start: {name} is not an uploaded raw tensor"
        ))),
    }
}

fn exl3<'w>(weights: &'w CudaWeights, name: &str) -> Result<&'w Exl3Weight, EngError> {
    match weights.get(name) {
        Some(CudaWeight::Exl3 { weight, .. }) => Ok(weight),
        _ => Err(EngError::Checkpoint(format!(
            "Flash-Next start: {name} is not an uploaded EXL3 matrix"
        ))),
    }
}

/// The trunk of the split MoE block of every layer, borrowed from the uploaded weights.
fn moe_trunks(
    weights: &'static CudaWeights,
    layers: usize,
) -> Result<Vec<SplitMoeTrunk<'static>>, EngError> {
    (0..layers)
        .map(|layer| {
            let at = |suffix: &str| layer_name(layer, suffix);
            Ok(SplitMoeTrunk {
                router: raw(weights, &at("mlp.gate.weight"))?,
                router_shared: raw(weights, &at("mlp.shared_expert_gate.weight"))?,
                shared_gate: exl3(weights, &at("mlp.shared_expert.gate_proj"))?,
                shared_up: exl3(weights, &at("mlp.shared_expert.up_proj"))?,
                shared_down: exl3(weights, &at("mlp.shared_expert.down_proj"))?,
            })
        })
        .collect()
}

/// A record source that adds up the time spent reading and packing records from the pack, so the
/// load log can split the bank fill into the pack read and the pin and upload.
struct TimedSource<'a> {
    inner: &'a dyn RecordSource,
    spent: Cell<Duration>,
    records: Cell<u64>,
}

impl RecordSource for TimedSource<'_> {
    fn record(&self, layer: usize, expert: u16, out: &mut [u8]) -> Result<(), EngError> {
        let opened = Instant::now();
        let result = self.inner.record(layer, expert, out);
        self.spent.set(self.spent.get() + opened.elapsed());
        self.records.set(self.records.get() + 1);
        result
    }

    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    fn upload(
        &self,
        layer: usize,
        expert: u16,
        ctx: &CudaContext,
        stream: &DeviceStream,
        dst: &DeviceBuffer,
        offset: usize,
    ) -> Result<bool, EngError> {
        let opened = Instant::now();
        let result = self.inner.upload(layer, expert, ctx, stream, dst, offset);
        self.spent.set(self.spent.get() + opened.elapsed());
        // Count only an ENGAGED upload: a `false` answer falls back to
        // `record`, which counts the expert there — once, either way.
        if matches!(result, Ok(true)) {
            self.records.set(self.records.get() + 1);
        }
        result
    }

    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    fn drain(&self, stream: &DeviceStream) -> Result<(), EngError> {
        self.inner.drain(stream)
    }
}

/// Reads an exllamav3 `expert_stats` JSON into per-layer expert use weights, in layer order.
/// Refuses a missing file, bad JSON, any layer key that is not `model.language_model.layers.<i>.mlp`,
/// or a layer index outside `0..layers`. A duplicate layer key cannot be detected here: the JSON
/// parser keeps the last value of a repeated object key, so the layer reads as missing instead.
/// The weights themselves are validated against the bank geometry by [`fill_from_profile`].
fn read_expert_stats(path: &str, layers: usize) -> Result<Vec<Vec<f64>>, EngError> {
    let bad =
        |detail: String| EngError::BadRequest(format!("expert_stats profile {path}: {detail}"));
    let text = std::fs::read_to_string(path).map_err(|e| bad(format!("unreadable: {e}")))?;
    let parsed: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| bad(format!("not valid JSON: {e}")))?;
    let layer_map = parsed
        .get("layers")
        .and_then(|v| v.as_object())
        .ok_or_else(|| bad("no layers object".into()))?;
    let mut out: Vec<Option<Vec<f64>>> = vec![None; layers];
    for (key, weights) in layer_map {
        let index: Option<usize> = key
            .strip_prefix("model.language_model.layers.")
            .and_then(|rest| rest.strip_suffix(".mlp"))
            .and_then(|n| n.parse().ok());
        let index = index.ok_or_else(|| bad(format!("layer key {key} is not expected")))?;
        if index >= layers {
            return Err(bad(format!("layer index {index} outside 0..{layers}")));
        }
        let row: Vec<f64> = serde_json::from_value(weights.clone())
            .map_err(|e| bad(format!("layer {index}: not a list of numbers: {e}")))?;
        if out[index].is_some() {
            return Err(bad(format!("layer {index} appears twice")));
        }
        out[index] = Some(row);
    }
    out.into_iter()
        .enumerate()
        .map(|(index, row)| row.ok_or_else(|| bad(format!("layer {index} is missing"))))
        .collect()
}

/// The per-layer slot fill: expert ids in slot order and their LRU counts.
type FillFromProfile = (Vec<Vec<i32>>, Vec<Vec<u64>>);

/// The slot rows and LRU counts of the initial fill. With a profile: per layer, the experts sorted
/// by use weight (highest first, id breaking ties) take the slots in rank order and the weights
/// scale to `u64` counts. Without: the first experts by id take the slots and lower ids count as
/// hotter. Pure; both arms keep every invariant `ReplaceOnMissLru::from_fill` checks.
fn fill_from_profile(
    geometry: BankGeometry,
    profile: Option<&[Vec<f64>]>,
) -> Result<FillFromProfile, EngError> {
    if let Some(profile) = profile
        && profile.len() != geometry.layers
    {
        return Err(EngError::BadRequest(format!(
            "Flash-Next expert bank: the popularity profile has {} layers, the bank has {}",
            profile.len(),
            geometry.layers
        )));
    }
    let mut rows = Vec::with_capacity(geometry.layers);
    let mut counts = Vec::with_capacity(geometry.layers);
    for layer in 0..geometry.layers {
        let Some(weights) = profile.and_then(|p| p.get(layer)) else {
            rows.push(
                (0..geometry.experts)
                    .map(|e| if e < geometry.slots { e as i32 } else { -1 })
                    .collect(),
            );
            counts.push((0..geometry.experts as u64).rev().collect());
            continue;
        };
        if weights.len() != geometry.experts || weights.iter().any(|w| !w.is_finite() || *w < 0.0) {
            return Err(EngError::BadRequest(format!(
                "Flash-Next expert bank: layer {layer} of the popularity profile has {} weights \
                 (need {}, each finite and non-negative)",
                weights.len(),
                geometry.experts
            )));
        }
        let mut by_weight: Vec<usize> = (0..geometry.experts).collect();
        by_weight.sort_by(|&a, &b| weights[b].total_cmp(&weights[a]).then(a.cmp(&b)));
        let mut row = vec![-1; geometry.experts];
        for (slot, &expert) in by_weight.iter().take(geometry.slots).enumerate() {
            row[expert] = slot as i32;
        }
        rows.push(row);
        counts.push(weights.iter().map(|w| (*w * 1000.0) as u64).collect());
    }
    Ok((rows, counts))
}

/// The bank: under the host plan the caller admitted. The plan's mode picks the store (an inclusive
/// plan holds every expert, so a miss copies one way only). `profile`, when the caller read one, is
/// the per-layer expert use weights of a serving run: the hottest `slots` experts fill the slots in
/// rank order and the counts keep them the last evicted; without one the first experts by id fill
/// the slots and lower ids count as hotter.
fn build_bank(
    projector: &CudaProjector,
    geometry: BankGeometry,
    plan: &HostPinPlan,
    profile: Option<&[Vec<f64>]>,
    usage_out: Option<&std::path::Path>,
    source: &dyn RecordSource,
    retained: Option<&'static crate::family::flash_next::cuda::bank_source::RungSource>,
) -> Result<DeviceBank, EngError> {
    let (rows, counts) = fill_from_profile(geometry, profile)?;
    let row_refs: Vec<&[i32]> = rows.iter().map(Vec::as_slice).collect();
    let policy = ReplaceOnMissLru::from_fill(geometry, &row_refs, &counts, usage_out)?;

    let state = ExpertBankState::with_policy(geometry, plan.mode, policy)?;
    let timed = TimedSource {
        inner: source,
        spent: Cell::new(Duration::ZERO),
        records: Cell::new(0),
    };
    let opened = Instant::now();
    let bank = DeviceBank::build(projector.context(), state, &timed, plan, retained)?;
    let total = opened.elapsed();
    eprintln!(
        "flash-next cuda: expert bank: {} records, pack read and pack {:.1} s, pin and upload {:.1} s, total {:.1} s",
        timed.records.get(),
        timed.spent.get().as_secs_f64(),
        total.saturating_sub(timed.spent.get()).as_secs_f64(),
        total.as_secs_f64()
    );
    Ok(bank)
}

#[allow(clippy::too_many_arguments)] // the values the start closure captured, passed once
fn make(
    stages: &Stages<'_>,
    artifact: &Path,
    wanted: &CudaSettings,
    announce: Option<std::sync::mpsc::SyncSender<Announcement<u32>>>,
    rules: CudaRules,
    identity: PrefixIdentity,
    prefill_chunk: u32,
    guard: BuildGuard,
) -> Result<(FlashCudaExecutor<StepDriver>, DecodeSession), EngError> {
    let driver = build_driver_resolving(
        &|name| stages.begin(name),
        artifact,
        wanted,
        announce.as_ref(),
        rules.sampler_vocab,
        prefill_chunk,
        Some(guard),
    );
    announce_failure(&driver, announce.as_ref());
    FlashCudaExecutor::new(driver?, rules, identity)
}

/// The figures one `tuning plan` line reports.
struct TuningCall<'a> {
    projector: &'a CudaProjector,
    free_vram: u64,
    vram_total: u64,
    experts: u64,
    record_bytes: u64,
    ring_budget: u64,
    ring_is_setting: bool,
    piece_rows: u64,
    row_window: u64,
    host_budget_bytes: u64,
    /// Streamed experts per layer; the loader's ring horizon is twice this.
    streamed_records: u64,
}

/// Prints the tuner's one `tuning plan` line: what it would pick beside
/// the values the start uses, every value with its source. A diagnostic
/// only — no value is applied, and a refused query degrades to zero
/// facts without latching the backend.
fn log_tuning_line(call: &TuningCall<'_>) {
    use crate::capability::tuning::{
        Decision, DeviceFacts, HardwareFacts, MemoryTopology, TuningInput, TuningMode,
        TuningOverrides, decide, log_line,
    };
    let (sm_count, compute_capability, pcie_link) =
        match call.projector.context().device_attributes() {
            Ok(attributes) => (
                attributes.sm_count,
                attributes.compute_capability,
                HardwareFacts::pcie_link_of(
                    &attributes.bus_id,
                    std::path::Path::new("/sys/bus/pci/devices"),
                ),
            ),
            Err(_) => (0, (0, 0), (0, 0)),
        };
    let (cpu_threads, cpu_isa) = HardwareFacts::host_side();
    let facts = HardwareFacts {
        topology: MemoryTopology::Discrete {
            host_budget_bytes: call.host_budget_bytes,
            devices: vec![DeviceFacts {
                vram_free_bytes: call.free_vram,
                vram_total_bytes: call.vram_total,
                sm_count,
                compute_capability,
                pcie_link,
            }],
        },
        cpu_threads,
        cpu_isa,
        drive: None,
    };
    // The ring candidate is the loader's own capped rule over the free
    // VRAM: whole expert records, never more than twice the streamed
    // experts; a dense model (no experts) offers none.
    let ring_candidates = if call.experts > 0 && call.record_bytes > 0 {
        let records = crate::family::flash_next::settings::ring_record_count(
            call.free_vram,
            call.record_bytes,
            2 * call.streamed_records,
        );
        vec![(
            records * call.record_bytes,
            "loader rule: whole records, capped at twice the streamed experts".to_string(),
        )]
    } else {
        Vec::new()
    };
    let decisions = [
        Decision {
            name: "piece",
            // No measured basis yet; the piece decision commit adds one.
            candidates: Vec::new(),
            in_use: (call.piece_rows, "resolved setting"),
        },
        Decision {
            name: "rows",
            // The row decision commit builds it from the prefill byte cost.
            candidates: Vec::new(),
            in_use: (call.row_window, "resolved setting"),
        },
        Decision {
            name: "ring",
            candidates: ring_candidates,
            in_use: (
                call.ring_budget,
                if call.ring_is_setting {
                    "setting"
                } else {
                    "default off"
                },
            ),
        },
    ];
    let mode = match std::env::var("INFERENCE_TUNER").ok().as_deref() {
        Some("off") => TuningMode::Off,
        _ => TuningMode::Auto,
    };
    let overrides = TuningOverrides::default();
    let input = TuningInput {
        mode,
        overrides: &overrides,
        facts: &facts,
        decisions: &decisions,
    };
    eprintln!("{}", log_line(&decide(&input), &input, false));
}

/// Loads the pack and builds the token driver with the context of `wanted`. `guard` pins the
/// one-engine flag at the first leak; a device test passes none.
pub(super) fn build_driver(
    stage: &dyn Fn(&'static str),
    artifact: &Path,
    wanted: &CudaSettings,
    sampler_vocab: usize,
    prefill_chunk: u32,
    guard: Option<BuildGuard>,
) -> Result<StepDriver, EngError> {
    build_driver_resolving(
        stage,
        artifact,
        wanted,
        None,
        sampler_vocab,
        prefill_chunk,
        guard,
    )
}

/// [`build_driver`], and when `announce` is given the context of `wanted` is a placeholder: the
/// memory plan picks it (see `default_context`), the start prints the choice and sends it
/// through `announce` for `start` (`await_announced`).
fn build_driver_resolving(
    stage: &dyn Fn(&'static str),
    artifact: &Path,
    wanted: &CudaSettings,
    announce: Option<&std::sync::mpsc::SyncSender<Announcement<u32>>>,
    sampler_vocab: usize,
    prefill_chunk: u32,
    mut guard: Option<BuildGuard>,
) -> Result<StepDriver, EngError> {
    super::driver::chunk::prof::check_switch_spelling()?;
    let refill_timing =
        super::driver::chunk::prof::check_switch_env("INFERENCE_FLASH_NEXT_DISK_PROFILE")?;
    let prefill_route_trace =
        super::driver::chunk::prof::check_switch_env("INFERENCE_FLASH_NEXT_PREFILL_ROUTE_TRACE")?;
    let exact_prefill_groups =
        super::driver::chunk::prof::check_switch_env("INFERENCE_FLASH_NEXT_PREFILL_EXACT_GROUPS")?;
    let prefill_waves = std::env::var_os("INFERENCE_FLASH_NEXT_PREFILL_WAVE_LAYER")
        .as_deref()
        .map(PrefillWaveScope::parse)
        .transpose()?;
    let began = Instant::now();
    stage("initializing CUDA");
    let projector = CudaProjector::new()?;
    register_served_kernels(projector.context())?;
    let card_memory = projector.memory_info()?;
    let free_vram = card_memory.free as u64;
    let vram_total = card_memory.total as u64;

    stage("opening the pack");
    let files = FlashPackFiles::open(artifact)?;
    let FlashPackFiles {
        checkpoint,
        ngram,
        report,
    } = files;
    let (model, kinds) = model_shape(&checkpoint.model_config)?;
    let layers = kinds.len();
    let ngram_bytes = std::fs::metadata(artifact.join(NGRAM_FILE))
        .map_err(|error| EngError::Checkpoint(format!("{NGRAM_FILE}: {error}")))?
        .len();

    stage("planning memory");
    let tier_shape = TierShape::from_pack(
        layers,
        model.moe.experts(),
        model.moe.top_k(),
        report.expert_bytes,
    )?;
    // One admission snapshot: `MemAvailable` clamped by the enclosing cgroup limits. The tier
    // plan, the host pin plan and the final guard all read this pair, so the plan that sizes the
    // store cannot build from a figure the guard disagrees with (a cgroup-capped host used to
    // plan an inclusive store the guard then refused at load time).
    let facts =
        Facts::read("Flash-Next expert bank", "Flash-Next")?.with_card(free_vram, vram_total);
    let host_admission = &facts.host;
    let (mem_total, mem_available) = host_admission.host_memory();
    let rows = wanted.prefill_rows as usize;
    let bulk_reader = ngram_bulk()? && rows > 1;
    let switches = crate::family::flash_next::settings::CacheSwitches::from_env()?;
    let router_proxy_enabled = switches.router_proxy;
    let disk_prefetch_horizon = switches.disk_prefetch_horizon;
    let join_active_prefetch = switches.join_active_prefetch;
    let projection_on_card = ple_device()?;
    let ple_shape = (projection_on_card && rows > 1).then(|| PleDeviceShape {
        width: model.hyper.hidden() * model.hyper.branches(),
        hidden: model.hyper.hidden(),
        embed: model.ple.embed(),
        rows_max: rows,
    });
    let ple_workspace = ple_shape
        .as_ref()
        .map_or(Ok(0), PleDeviceShape::workspace_bytes)?;
    let batch_switch = ple_batch()?;
    // The prefill ring's budget, carved out before the load plan so the bank sizes itself
    // into what remains: at most 1.5 GiB and 30% of the free card, only for chunked prefill.
    // A ring below one streamed layer is not built (the multi-row path then keeps its
    // union-admit shape, correct and slower).
    let record_bytes = PackRecordSource::new(&checkpoint)?.layout().record_bytes();
    // INFERENCE_FLASH_NEXT_RING_BUDGET sets the ring's bytes. The ring measures a serve
    // regression today (it re-issues its copies every chunk while the union-admit path
    // copies once), so the default is off; the flag keeps the lane's experiments open.
    // A budget above the free VRAM is clamped, not wrapped: the plan then fits the card.
    let ring_spec = std::env::var("INFERENCE_FLASH_NEXT_RING_BUDGET").ok();
    let ring_budget = crate::family::flash_next::settings::ring_budget_bytes(ring_spec.as_deref())?;
    // The tuner's line prints after the ring build below, so its ring
    // horizon is the loader's own, not an independent guess.

    // `FlashLoadPlan::new` replaces the trunk and state bytes (the `(0, 0)`) and the K/V (the
    // placeholder positions) with the numbers the pack and the context give. The shared prefill
    // arena serves chunked prefill only; `prefill_rows = 1` is the per-token driver and keeps
    // the pre-switch charge of zero. The arena grows with the context, so the plan is built per
    // context.
    let geometry_for_arena = FlashGeometry::new(&model, &kinds)?;
    let plan_for = |context: u32| -> Result<FlashLoadPlan, EngError> {
        let prefill_arena_bytes = if rows > 1 {
            StepPlan::prefill_bytes(true, &geometry_for_arena, rows, context as usize)?
        } else {
            0
        };
        let card = TierInputs::measured(
            tier_shape,
            (0, 0),
            KvPlan {
                positions: 1,
                bytes_per_position: 1,
            },
            free_vram.saturating_sub(ring_budget),
            (mem_total, mem_available),
            // The same infallible placeholder the two-phase choice below charges
            // first: the pooled depth's records in flight, from the one shared
            // constant. The real figure is read only when the tier already points
            // at the disk rung, and re-validated there.
            rung_ring_placeholder_bytes(record_bytes),
        )
        .with_prefill_arena(prefill_arena_bytes)?
        .with_prefill_arena(ple_workspace)?;
        FlashLoadPlan::new(&report, &checkpoint.model_config, ngram_bytes, card)
    };
    let kv_format = wanted.kv_format.format();
    let min_slots = model.moe.top_k() + 1;
    let resolved;
    let wanted = if let Some(announce) = announce {
        let floor = min_slots.max(wanted.slots.map_or(0, |asked| asked as usize));
        let slots_at = |context: u32| {
            plan_for(context)
                .and_then(|plan| plan.at_context(u64::from(context), kv_format, floor))
                .ok()
                .map(|load| load.context.plan.slots_per_layer())
        };
        let cap = crate::family::flash_next::settings::max_positions(&checkpoint.model_config)
            .unwrap_or(crate::family::flash_next::settings::MAX_CUDA_CONTEXT)
            .min(crate::family::flash_next::settings::MAX_CUDA_CONTEXT);
        let chosen = crate::family::flash_next::settings::default_context(&slots_at, floor, cap);
        let context = match &chosen {
            Some(chosen) => {
                eprintln!(
                    "flash-next cuda: context {} chosen from the memory plan: {} expert slots per \
                     layer at {}, {} at {} (model maximum {}); set flash_next.context to override",
                    chosen.context,
                    chosen.slots,
                    chosen.context,
                    chosen.baseline_slots,
                    crate::family::flash_next::settings::DEFAULT_CUDA_CONTEXT,
                    cap
                );
                chosen.context
            }
            None => {
                eprintln!(
                    "flash-next cuda: the memory plan admits no context from the default {}; \
                     the load reports the shortfall",
                    wanted.context
                );
                wanted.context
            }
        };
        // A closed channel means `start` has given up; the build then fails on its own terms.
        let _ = announce.send(Ok(context));
        resolved = CudaSettings {
            context,
            ..wanted.clone()
        };
        &resolved
    } else {
        wanted
    };
    let load_plan = plan_for(wanted.context)?;
    let load = load_plan.at_context(u64::from(wanted.context), kv_format, min_slots)?;
    let planned_slots = load.context.plan.slots_per_layer();
    let slots = match wanted.slots {
        None => planned_slots,
        Some(asked) if (asked as usize) <= planned_slots => asked as usize,
        Some(asked) => {
            return Err(EngError::MemoryPlan(format!(
                "Flash-Next start: slots {asked} per layer asked, the memory plan at context {} leaves {planned_slots}",
                wanted.context
            )));
        }
    };
    eprintln!("{load}; serving {slots} slots per layer");
    if sampler_vocab == 0 || sampler_vocab > model.vocab {
        return Err(EngError::Checkpoint(format!(
            "sampler vocabulary {} is outside 1..={}, the width of the model head",
            sampler_vocab, model.vocab
        )));
    }

    // The host side is admitted before anything is uploaded: the pinned store of the bank, the
    // engine heap and margin of `HostTerms`, including any host F32 PLE projections.
    let ple_width = (model.hyper.hidden() * model.hyper.branches() + model.hyper.hidden()) as u64;
    let ple_bytes = if ple_shape.is_none() {
        ple_width * model.ple.embed() as u64 * 4
    } else {
        0
    };
    let prefetch_backing = if disk_prefetch_horizon.is_some() {
        bank_source::RungSource::backing_bytes(&checkpoint, artifact, 1)?
    } else {
        0
    };
    // One raw slot, the worker's aligned read window, and the packed records in
    // the current upload and the pending layers. Include the transient pack workspace.
    let prefetch_bytes = disk_prefetch_horizon.map_or(0, |horizon| {
        let records = (horizon + 1) * super::disk_prefetch::CANDIDATES + 2;
        2 * prefetch_backing as u64 + records as u64 * record_bytes as u64
    });
    let victim_budget = switches.victim_cache_bytes;
    let victim_layout = (victim_budget > 0)
        .then(|| {
            super::victim_cache::VictimLayout::new(victim_budget, model.moe.top_k(), record_bytes)
        })
        .transpose()?;
    let selective_victims = switches.selective_victims;
    let recent_bytes = if selective_victims {
        super::victim_cache::RecentPicks::bytes(layers, model.moe.experts())
    } else {
        0
    };
    let victim_bytes = victim_layout
        .as_ref()
        .map_or(0, |layout| layout.bytes() as u64);
    let engine_heap = (HostTerms::DEFAULT.engine_heap + ple_bytes + prefetch_bytes)
        .checked_add(victim_bytes)
        .and_then(|bytes| bytes.checked_add(recent_bytes))
        .ok_or_else(|| {
            EngError::MemoryPlan("Flash-Next victim cache host budget overflows".into())
        })?;
    let terms = HostTerms {
        engine_heap,
        ..HostTerms::DEFAULT
    };
    let geometry = BankGeometry::new(
        layers,
        model.moe.experts(),
        model.moe.top_k(),
        slots,
        record_bytes,
    )?;
    // The store tier: the shared ladder takes inclusive while the one host budget carries it,
    // then the exclusive store of spare places (exactly today's auto), then the disk tier —
    // no pinned store, every cold record read through the staging ring on demand (the disk
    // rung; below ring + overhead + cache floor the refusal stays, the honest minimum tier).
    // `FIFI_FLASH_NEXT_BANK_MODE` forces a tier (`inclusive`, `exclusive` or `disk`);
    // anything else is refused, never a silent fallback.
    let memory = HostMemory {
        total: mem_total,
        available: mem_available,
    };
    // The rung's pooled read depth (RUNG_RING_SLOTS, in the shared tier
    // module): one worker per slot, one RECORD per job. The span bench's
    // step cells issue two records per layer, so it cannot separate depth 4
    // from depth 8; 4 is provisional.
    // The staging ring's own backing — the disk rung's read path and its budget floor.
    // Sizing is two-phase so every non-disk start stays byte-for-byte the pre-rung start:
    // the first pass charges the infallible placeholder (the full pooled depth's
    // records in flight) and never touches the pack; only when that already points at
    // the disk tier is the real figure read from the ring source and the choice made
    // once more with it — so the ladder, the disk plan and the source that serves the
    // rung all charge one number (the allocated slots), and a non-disk answer is never
    // revised.
    let mode = std::env::var("FIFI_FLASH_NEXT_BANK_MODE").ok();
    let ring_placeholder = rung_ring_placeholder_bytes(record_bytes);
    let tier = HostPinPlan::bank_mode_tier(
        mode.as_deref(),
        &geometry,
        &memory,
        &terms,
        ring_placeholder,
    )?;
    if disk_prefetch_horizon.is_some() && !matches!(tier, TierChoice::Disk) {
        return Err(EngError::BadRequest(
            "INFERENCE_FLASH_NEXT_DISK_PREFETCH needs the Flash-Next disk tier".into(),
        ));
    }
    if victim_layout.is_some() && !matches!(tier, TierChoice::Disk) {
        return Err(EngError::BadRequest(
            "INFERENCE_FLASH_NEXT_DISK_VICTIM_CACHE_BYTES needs the Flash-Next disk tier".into(),
        ));
    }
    let (tier, ring_bytes) = match tier {
        TierChoice::Disk => {
            let real =
                bank_source::RungSource::backing_bytes(&checkpoint, artifact, RUNG_RING_SLOTS)?
                    as u64;
            let tier =
                HostPinPlan::bank_mode_tier(mode.as_deref(), &geometry, &memory, &terms, real)?;
            if !matches!(tier, TierChoice::Disk) {
                return Err(EngError::MemoryPlan(format!(
                    "the disk tier's real ring backing ({real} B) no longer fits where the \
                     {RUNG_RING_SLOTS}-record placeholder did"
                )));
            }
            (tier, real)
        }
        other => (other, ring_placeholder),
    };
    let host_plan = match tier {
        TierChoice::Inclusive => HostPinPlan::new(&geometry, HostMode::Inclusive, &memory, &terms)?,
        TierChoice::Exclusive { spare_places: 1 } => {
            HostPinPlan::new(&geometry, HostMode::Exclusive, &memory, &terms)?
        }
        TierChoice::Exclusive { spare_places } => {
            return Err(EngError::MemoryPlan(format!(
                "an exclusive store of {spare_places} spare places per layer is not wired: \
                 today's exclusive store pins exactly one"
            )));
        }
        TierChoice::Disk => HostPinPlan::new_disk(&geometry, &memory, &terms, ring_bytes)?,
        TierChoice::CardTiered { .. } => {
            return Err(EngError::MemoryPlan(
                "the host ladder answered a card tier to a start that passed no card figures"
                    .into(),
            ));
        }
    };
    let (tier_name, speed_class) = tier.describe();
    // The rung reader is chosen and its read mode decided ONCE, here —
    // before the trunk upload — so a filesystem that refuses O_DIRECT
    // serves buffered instead of failing the start after the trunk is
    // already loaded. Strict spelling, like the resume switch.
    let ring_reader = match std::env::var("FIFI_FLASH_NEXT_PACK_RING") {
        Ok(value) if value == "1" => true,
        Err(std::env::VarError::NotPresent) => matches!(tier, TierChoice::Disk),
        Ok(_) | Err(std::env::VarError::NotUnicode(_)) if matches!(tier, TierChoice::Disk) => true,
        Ok(other) => {
            return Err(EngError::BadRequest(format!(
                "FIFI_FLASH_NEXT_PACK_RING must be 1 or unset, found {other:?}"
            )));
        }
        Err(std::env::VarError::NotUnicode(value)) => {
            return Err(EngError::BadRequest(format!(
                "FIFI_FLASH_NEXT_PACK_RING must be 1 or unset, found {value:?}"
            )));
        }
    };
    let rung_mode = if ring_reader {
        Some(bank_source::RungSource::probe_read_mode(
            &checkpoint,
            artifact,
        )?)
    } else {
        None
    };
    let rung_word = match &rung_mode {
        Some(mode) => format!(", rung mode: {}", mode.plain()),
        None => String::new(),
    };
    eprintln!(
        "flash-next cuda: {tier_name} ({speed_class}){rung_word}, {} pinned host bytes",
        host_plan.store_bytes
    );
    // The guard reads the same snapshot the plan sized from, and the PLE bytes already ride
    // inside `process_bytes` (the `engine_heap` term), so plan and guard admit one figure.
    host_admission.admit_for(
        host_plan.process_bytes,
        "Flash-Next expert bank and PLE projections",
    )?;

    // LEAK: the owners are leaked for the process lifetime, giving the step `'static` borrows.
    // From the first leak a failure leaves the card loaded, so the one-engine flag is pinned here.
    let projector: &'static CudaProjector = Box::leak(Box::new(projector));
    let checkpoint: &'static Checkpoint = Box::leak(Box::new(checkpoint));
    if let Some(guard) = guard.as_mut() {
        guard.commit();
    }
    let ctx = projector.context();

    stage("loading weights");
    let list = TrunkList::new(&checkpoint.source, &checkpoint.model_config)?;
    let reader = PackReader::new(projector, checkpoint, Reserve::default());
    let weights: &'static CudaWeights =
        Box::leak(Box::new(trunk_weights(&reader, &list, &load_plan)?));
    eprintln!(
        "flash-next cuda: trunk uploaded in {:.1} s",
        began.elapsed().as_secs_f64()
    );

    stage("filling the expert bank");
    // The record source: the buffered pack reader, or — behind
    // FIFI_FLASH_NEXT_PACK_RING=1, default off — the pooled ring reader,
    // decided with its read mode at the tier line above. The disk tier
    // forces the ring reader regardless of the switch: the rung's
    // every-miss read IS the ring.
    let layout: crate::family::flash_next::moe::slot_bank::SlotBankLayout;
    enum Source {
        Buffered(Box<dyn RecordSource>),
        Rung(Box<bank_source::RungSource>),
    }
    let source = if ring_reader {
        let bytes = bank_source::RungSource::backing_bytes(checkpoint, artifact, RUNG_RING_SLOTS)?;
        let backing: &'static mut [u8] = Box::leak(vec![0u8; bytes].into_boxed_slice());
        // The serve-state pattern of the exl3 bank: register the plain
        // bytes so the H2D from the slot windows is truly async. Best-
        // effort — an unregistered window still answers correctly, it
        // only loses the async copy.
        match host_register(backing) {
            Ok(()) => {}
            Err(error) => eprintln!(
                "flash-next cuda: ring unpinned, staged copies (host_register refused: {error:?})"
            ),
        }
        let mode = rung_mode.expect("the ring reader was probed at the tier line");
        let staged =
            bank_source::RungSource::new(checkpoint, artifact, RUNG_RING_SLOTS, mode, backing)?;
        layout = staged.layout().clone();
        eprintln!(
            "flash-next cuda: record source: the pooled ring store, {}",
            staged.read_mode().plain()
        );
        Source::Rung(Box::new(staged))
    } else {
        let staged = bank_source::PackRecordSource::new(checkpoint)?;
        layout = staged.layout().clone();
        Source::Buffered(Box::new(staged))
    };
    // LEAK: only the disk rung keeps its reader for the process lifetime — its every-miss
    // reads outlive this start, and the `static borrow reaches the step. Every other tier
    // fills from the source at the start below and drops it with this call, byte-for-byte
    // the pre-rung start.
    let (fill_source, retained_source): (
        &dyn RecordSource,
        Option<&'static bank_source::RungSource>,
    ) = if matches!(tier, TierChoice::Disk) {
        let Source::Rung(boxed) = source else {
            // The disk tier forces the ring reader above, so this arm
            // cannot happen; refuse rather than leak the wrong type.
            return Err(EngError::BadRequest(
                "the disk tier requires the ring reader".into(),
            ));
        };
        let leaked: &'static bank_source::RungSource = Box::leak(boxed);
        (leaked, Some(leaked))
    } else {
        // The fill borrows the enum: either source drops with this call,
        // byte-for-byte the pre-rung start.
        let fill: &dyn RecordSource = match &source {
            Source::Rung(rung) => rung.as_ref(),
            Source::Buffered(buffered) => buffered.as_ref(),
        };
        (fill, None)
    };
    // Optional popularity profile, an exllamav3 expert_stats JSON:
    // {"model_dir": .., "layers": {"model.language_model.layers.<i>.mlp": [weights]}}. Sources,
    // in order: the engine's `expert_stats` key (which also takes the usage dump, so the next
    // start of the same entry serves warm) and `FIFI_FLASH_NEXT_EXPERT_STATS`. A set env value
    // must read and parse; a malformed file is refused, never a silent identity fill. A
    // not-yet-written engine-key file is a cold start (the dump creates it on shutdown).
    let env_stats = std::env::var("FIFI_FLASH_NEXT_EXPERT_STATS")
        .ok()
        .filter(|path| !path.is_empty());
    let stats_path = wanted
        .expert_stats
        .as_deref()
        .map(std::path::Path::new)
        .or_else(|| env_stats.as_deref().map(std::path::Path::new));
    let profile = match &stats_path {
        None => None,
        Some(path) if wanted.expert_stats.is_some() && !path.exists() => {
            eprintln!(
                "flash-next cuda: expert bank: no profile at {} yet; identity fill, the dump writes it on shutdown",
                path.display()
            );
            None
        }
        Some(path) => Some(read_expert_stats(&path.display().to_string(), layers)?),
    };
    let usage_out = wanted.expert_stats.clone().or_else(|| {
        std::env::var("FIFI_FLASH_NEXT_EXPERT_USAGE_OUT")
            .ok()
            .filter(|path| !path.is_empty())
            .map(std::path::PathBuf::from)
    });
    if profile.is_some() {
        eprintln!(
            "flash-next cuda: expert bank: popularity profile {}",
            stats_path.map_or(String::new(), |p| p.display().to_string())
        );
    }
    let mut bank = build_bank(
        projector,
        geometry,
        &host_plan,
        profile.as_deref(),
        usage_out.as_deref(),
        fill_source,
        retained_source,
    )?;
    if refill_timing {
        bank.enable_refill_timing();
    }
    if prefill_route_trace {
        bank.enable_prefill_route_trace();
    }
    if exact_prefill_groups {
        bank.enable_exact_prefill_groups();
        if bank.uses_exact_prefill_groups() {
            eprintln!("flash-next cuda: exact disk prefill groups enabled");
        }
    }
    if let Some(scope) = prefill_waves {
        bank.enable_prefill_waves(scope)?;
        match scope {
            PrefillWaveScope::Layer(layer) => {
                eprintln!("flash-next cuda: disk prefill expert waves enabled at layer {layer}");
            }
            PrefillWaveScope::All => {
                eprintln!("flash-next cuda: disk prefill expert waves enabled at all layers");
            }
        }
    }
    eprintln!(
        "flash-next cuda: expert bank filled at {:.1} s",
        began.elapsed().as_secs_f64()
    );

    if let Some(horizon) = disk_prefetch_horizon {
        let backing = Box::leak(vec![0; prefetch_backing].into_boxed_slice());
        bank.enable_disk_prefetch(backing, horizon, join_active_prefetch)?;
        eprintln!(
            "flash-next cuda: disk prefetch N+{horizon} enabled, host reserve {prefetch_bytes} bytes, join={join_active_prefetch}"
        );
    }

    if let Some(layout) = &victim_layout {
        bank.enable_victim_cache(ctx, layout, selective_victims)?;
        eprintln!(
            "flash-next cuda: disk victim cache {} records, {} pinned bytes including upload reserve",
            layout.resident_records(),
            layout.bytes(),
        );
        if selective_victims {
            eprintln!(
                "flash-next cuda: selective victim write-back enabled, {recent_bytes} history bytes"
            );
        }
    }

    stage("building the step");
    // `prefill_rows` above one builds the step's plans at that width, so a prompt chunk runs
    // as device chunks of that many rows; one keeps the per-token prefill. The default
    // width is 256; the supported range reaches `exl3_rows_ceiling()`.
    let stream = Box::leak(Box::new(ctx.stream()?));
    let new_step = if rows > 1 {
        FlashStep::new_prefill
    } else {
        FlashStep::new
    };
    let step = new_step(
        ctx,
        stream,
        weights,
        &checkpoint.model_config,
        wanted.context as usize,
        rows,
        0..layers,
    )?;
    // The streamed experts per layer — the loader's ring horizon is twice
    // this, and the tuner's line below reports the same capped rule.
    let bank_geometry = *bank.geometry();
    let streamed = bank_geometry
        .experts
        .checked_sub(bank_geometry.slots)
        .unwrap_or(bank_geometry.experts) as u64;
    let mut block = SplitMoe::new(
        ctx,
        moe_trunks(weights, layers)?,
        bank,
        &layout,
        &step.moe_dims(rows),
    )?;
    if router_proxy_enabled || disk_prefetch_horizon.is_some() {
        block.enable_router_proxy(ctx, model.hyper.hidden(), router_proxy_enabled)?;
    }
    if router_proxy_enabled {
        eprintln!("flash-next cuda: future-router probe enabled");
    }
    // The prefill streaming ring refills its slots from the bank's store (demotion); the
    // disk rung has no store to stream from, so funding the ring under it would strand the
    // slots — every whole-chunk expert serves as a no-ring-entry through the rung's reads.
    let ring_reason = crate::family::flash_next::settings::CacheSwitches::ring_unfunded_reason(
        rows,
        ring_budget,
        matches!(host_plan.mode, HostMode::Disk),
    );
    switches.require_funded_ring(ring_reason)?;
    if ring_reason.is_none() {
        let uploads = switches.ring;
        let slots = block.plan_ring(
            ctx,
            crate::family::flash_next::settings::ring_record_count(
                ring_budget,
                record_bytes as u64,
                2 * streamed,
            ) as usize,
            wanted.ring_rows as usize,
            uploads,
        )?;
        if slots == 0 {
            switches.require_funded_ring(Some("the ring budget cannot hold one full layer"))?;
            eprintln!(
                "flash-next cuda: prefill ring not funded ({} bytes reserved); the union-admit \
             path serves prefill",
                ring_budget
            );
        } else {
            eprintln!(
                "flash-next cuda: prefill ring of {slots} slots ({:.2} GiB) behind the bank; \
                 shared prefill whole-chunk at {} rows and above, {} uploads",
                slots as f64 * record_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                wanted.ring_rows,
                uploads.label()
            );
        }
    }
    // The tuner's line, after the ring build so its ring horizon is the
    // loader's own (twice the streamed experts). A diagnostic only; no
    // value is applied.
    log_tuning_line(&TuningCall {
        projector,
        free_vram,
        vram_total,
        experts: model.moe.experts() as u64,
        record_bytes: record_bytes as u64,
        ring_budget,
        ring_is_setting: ring_spec.is_some(),
        // The piece is the device chunk (16 by default); the window is
        // the operator-reachable row width (to 2,048).
        piece_rows: prefill_chunk as u64,
        row_window: wanted.prefill_rows as u64,
        host_budget_bytes: mem_available,
        streamed_records: streamed,
    });
    let model_layers: Vec<usize> = (0..layers).collect();
    let moe = SplitMoeAt::new(block, &model_layers)?;

    stage("opening the n-gram table");
    let ngram_rows = NgramRows::new(ngram)?;
    let reader = bulk_reader
        .then(|| ngram_rows.reader(DEFAULT_THREADS))
        .transpose()?;
    let device_ple = if let Some(shape) = ple_shape {
        stage("allocating the PLE projection workspace");
        let layer = model.ple_layer;
        let (key, key_dtype) = step.half(&layer_name(layer, "ple.key_proj.weight"))?;
        let (value, value_dtype) = step.half(&layer_name(layer, "ple.value_proj.weight"))?;
        if key_dtype != value_dtype {
            return Err(EngError::Checkpoint(
                "the PLE projection matrices have different dtypes".into(),
            ));
        }
        let dtype = match key_dtype {
            HyperDtype::F16 => Dtype::F16,
            HyperDtype::Bf16 => Dtype::Bf16,
        };
        Some(DevicePle::new(ctx, dtype, [key, value], shape)?)
    } else {
        if projection_on_card {
            eprintln!(
                "flash-next cuda: INFERENCE_FLASH_NEXT_PLE_DEVICE ignored (prefill rows = 1)"
            );
        }
        None
    };
    // The batch host route (chunked prefill only, no device route): the feed takes precomputed
    // pairs from one blocked product per chunk; the projection runs that product.
    let batch_ple = match batch_switch.then_some(rows > 1) {
        Some(true) if device_ple.is_none() => true,
        Some(_) => {
            if device_ple.is_none() {
                eprintln!(
                    "flash-next cuda: INFERENCE_FLASH_NEXT_PLE_BATCH ignored (prefill rows = 1)"
                );
            } else {
                eprintln!(
                    "flash-next cuda: INFERENCE_FLASH_NEXT_PLE_BATCH ignored (PLE_DEVICE route)"
                );
            }
            false
        }
        None => false,
    };
    let (ple, ple_host_batch) = if device_ple.is_some() {
        let vectors = PleVectors::read_vectors(&checkpoint.source, model.ple_layer, &model.ple)?;
        (
            PleFeed::with_fixed(
                model.ple,
                ngram_rows,
                vectors,
                Rc::new(FixedProjection::new()),
            ),
            None,
        )
    } else {
        let (vectors, projection) = PleVectors::read(
            &checkpoint.source,
            model.ple_layer,
            &model.ple,
            ple_threads(),
        )?;
        if batch_ple {
            (
                PleFeed::with_fixed(
                    model.ple,
                    ngram_rows,
                    vectors,
                    Rc::new(FixedProjection::new()),
                ),
                Some(Box::new(projection)),
            )
        } else {
            (
                PleFeed::new(model.ple, ngram_rows, vectors, Box::new(projection)),
                None,
            )
        }
    };
    let hidden = model.hyper.hidden();
    let branches = model.hyper.branches();
    let ple_embed = model.ple.embed();
    let feeds = PackFeeds::new(&checkpoint.source, ple, hidden, branches, reader);
    let shape = DriverShape {
        layers,
        ple_layer: model.ple_layer,
        positions: wanted.context as usize,
        start: StartShape {
            sampler_vocab,
            stream: branches * hidden,
            ple_embed,
        },
    };
    let driver = if rows > 1 {
        let ple_route = device_ple;
        let ple_on_card = ple_route.is_some();
        let device = CardChunkDevice::with_ple(step, moe, ple_route);
        let mut chunked = ChunkDriver::new(device, feeds, shape)?;
        chunked.chunk_cap = rows;
        chunked.ple_device = ple_on_card;
        let route_note = if ple_on_card {
            ", PLE projections on the card"
        } else if ple_host_batch.is_some() {
            ", PLE projections batched on the host"
        } else {
            ""
        };
        chunked.ple_host_batch = ple_host_batch;
        eprintln!(
            "flash-next cuda: ready in {:.1} s, prefill chunks of {rows} rows{route_note}",
            began.elapsed().as_secs_f64(),
        );
        StepDriver::Chunk(chunked)
    } else {
        eprintln!(
            "flash-next cuda: ready in {:.1} s",
            began.elapsed().as_secs_f64()
        );
        StepDriver::Token(TokenDriver::new(CardDevice::new(step, moe), feeds, shape)?)
    };
    Ok(driver)
}

/// The one driver type at the engine boundary: the plain token driver of decode, or the chunk
/// driver `prefill_rows` builds. Forwards the stepper and the device-test surface.
pub(in crate::family::flash_next::cuda) enum StepDriver {
    Token(TokenDriver<CardDevice, PackFeeds>),
    Chunk(ChunkDriver<CardChunkDevice, PackFeeds>),
}

impl crate::family::session::TokenStepper for StepDriver {
    const OPERATION: &'static str = "flash_next_cuda";

    fn feed(&mut self, token: u32) -> Result<(), EngError> {
        match self {
            Self::Token(driver) => driver.feed(token),
            Self::Chunk(driver) => driver.feed(token),
        }
    }

    /// Forwarded, not defaulted: the chunk driver's whole point is the chunked `feed_rows`.
    fn feed_rows(&mut self, tokens: &[u32]) -> Result<(), EngError> {
        match self {
            Self::Token(driver) => driver.feed_rows(tokens),
            Self::Chunk(driver) => driver.feed_rows(tokens),
        }
    }

    fn logits(&mut self, token: u32) -> Result<Vec<f32>, EngError> {
        match self {
            Self::Token(driver) => driver.logits(token),
            Self::Chunk(driver) => driver.logits(token),
        }
    }

    fn fed(&self) -> usize {
        match self {
            Self::Token(driver) => driver.fed(),
            Self::Chunk(driver) => driver.fed(),
        }
    }

    fn reset(&mut self) -> Result<(), EngError> {
        match self {
            Self::Token(driver) => driver.reset(),
            Self::Chunk(driver) => driver.reset(),
        }
    }
}

impl StepperPositions for StepDriver {
    fn positions(&self) -> usize {
        match self {
            Self::Token(driver) => driver.positions(),
            Self::Chunk(driver) => driver.positions(),
        }
    }
}

impl ResumesState for StepDriver {
    fn resume_generation(&self) -> u64 {
        match self {
            Self::Token(driver) => driver.resume_generation(),
            Self::Chunk(driver) => driver.resume_generation(),
        }
    }

    fn bank_snapshot(&self) -> Option<crate::sched::job::PrefillBankReport> {
        StepDriver::bank_snapshot(self)
    }

    fn rung_miss_line(&self) -> Option<String> {
        match self {
            Self::Token(driver) => driver.rung_miss_line(),
            Self::Chunk(driver) => driver.rung_miss_line(),
        }
    }

    fn discard_decode_step(&self) {
        match self {
            Self::Token(driver) => driver.discard_decode_step(),
            Self::Chunk(driver) => driver.discard_decode_step(),
        }
    }

    fn resume_host_bytes(&self) -> u64 {
        match self {
            Self::Token(driver) => driver.resume_host_bytes(),
            Self::Chunk(driver) => driver.resume_host_bytes(),
        }
    }

    fn save_resume(&self) -> Result<ResumeSnapshot, EngError> {
        match self {
            Self::Token(driver) => driver.save_resume(),
            Self::Chunk(driver) => driver.save_resume(),
        }
    }

    fn restore_resume(&mut self, snapshot: &ResumeSnapshot) -> Result<(), EngError> {
        match self {
            Self::Token(driver) => driver.restore_resume(snapshot),
            Self::Chunk(driver) => driver.restore_resume(snapshot),
        }
    }
}

impl StepDriver {
    /// The bank's miss count for the per-job report.
    pub(super) fn bank_misses(&self) -> u64 {
        match self {
            Self::Token(driver) => driver.device().bank_misses(),
            Self::Chunk(driver) => driver.device().bank_misses(),
        }
    }

    /// The bank's lifetime counters for the request log's `prefill_bank_*` keys (log-only):
    /// the counters of whichever driver mode holds the device.
    pub(super) fn bank_snapshot(&self) -> Option<crate::sched::job::PrefillBankReport> {
        match self {
            Self::Token(driver) => ResumesState::bank_snapshot(driver),
            Self::Chunk(driver) => ResumesState::bank_snapshot(driver),
        }
    }

    /// The single-row kernel counters, for the device tests.
    #[cfg(test)]
    pub(super) fn kernel_launches(&self) -> crate::kernels::single_row::SingleRowLaunches {
        match self {
            Self::Token(driver) => driver.device().kernel_launches(),
            Self::Chunk(driver) => driver.device().kernel_launches(),
        }
    }

    /// The single-row kernel name, for the device tests (both devices expose it).
    #[cfg(test)]
    pub(super) fn single_row_kernel_name(&self) -> Result<&'static str, EngError> {
        match self {
            Self::Token(driver) => driver.device().single_row_kernel_name(),
            Self::Chunk(driver) => driver.device().single_row_kernel_name(),
        }
    }

    /// The bank's placement digest (slots per layer, SHA-256 of the resident set at load):
    /// the state a run record's step-0 logits hash compares at.
    #[cfg(test)]
    pub(super) fn placement_digest(&self) -> (u32, String) {
        match self {
            Self::Token(driver) => driver.device().placement_digest(),
            Self::Chunk(driver) => driver.device().placement_digest(),
        }
    }

    /// The last step's stage times, for the driver device tests. The chunk driver does not
    /// track stage times; the stage table runs on the token driver.
    pub(super) fn last_times(&self) -> TokenTimes {
        match self {
            Self::Token(driver) => driver.last_times(),
            Self::Chunk(_) => TokenTimes::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BankGeometry, HostMemory, HostPinPlan, HostTerms, fill_from_profile, read_expert_stats,
    };
    use std::io::Write as _;

    /// The Disk rung's refusals carry the plan's own byte counts, computed here from the same
    /// terms the plan uses: the floor refusal names every term, and the page-cache refusal names
    /// its leftover — both asserted exactly, never by substring.
    #[test]
    fn the_disk_rung_refuses_below_its_own_floor_with_the_plans_counts() {
        let geometry = BankGeometry::new(1, 6, 2, 3, 8).expect("geometry");
        let terms = HostTerms::DEFAULT;
        let memory = HostMemory {
            total: 1 << 30,
            available: 1,
        };
        let ring = 4096u64;
        let scratch = 2 * (8 + terms.stream_slot_metadata);
        let staging = terms.ngram_staging_bytes().expect("the staging bytes");
        let os_floor = memory.os_floor();
        let process = ring + scratch + staging + terms.engine_heap + terms.margin;
        let needed = process + os_floor;
        let error = HostPinPlan::new_disk(&geometry, &memory, &terms, ring)
            .expect_err("one available byte cannot carry the ring plus the floor");
        assert_eq!(
            error.describe(),
            format!(
                "Flash-Next expert bank: {ring} staging-ring bytes, {scratch} stream scratch, \
                 {staging} n-gram staging, \
                 {} heap and {} margin plus the {os_floor} byte OS floor need {needed} bytes, \
                 {} are available",
                terms.engine_heap, terms.margin, memory.available
            )
        );
        // The page-cache refusal: the floor fits, one byte of the cache floor does not.
        let memory = HostMemory {
            total: 1 << 30,
            available: needed + (4u64 << 30) - 1,
        };
        let terms = HostTerms {
            ngram_cache_min: 4u64 << 30,
            ..HostTerms::DEFAULT
        };
        let error = HostPinPlan::new_disk(&geometry, &memory, &terms, ring)
            .expect_err("the page-cache floor is short by one byte");
        assert_eq!(
            error.describe(),
            format!(
                "Flash-Next expert bank: {} bytes of page cache are left of {} available; \
                 the n-gram page-cache floor is {}",
                terms.ngram_cache_min - 1,
                memory.available,
                terms.ngram_cache_min
            )
        );
    }

    /// A profile fills the slots in weight rank order (ties by id) and scales the weights to
    /// counts; the same bank without one keeps the id-ordered fill and the reversed-id counts.
    #[test]
    fn a_profile_ranks_the_initial_fill() {
        let geometry = BankGeometry::new(1, 6, 2, 3, 8).expect("geometry");
        let profile = [vec![1.0, 9.0, 4.0, 9.0, 0.0, 2.0]];
        let (rows, counts) = fill_from_profile(geometry, Some(&profile)).expect("fill");
        // experts 1 and 3 tie at the top weight; id breaks it, so 1 takes slot 0, 3 slot 1, 2 slot 2.
        assert_eq!(rows[0], [-1, 0, 2, 1, -1, -1]);
        assert_eq!(counts[0], vec![1000, 9000, 4000, 9000, 0, 2000]);
        let (rows, counts) = fill_from_profile(geometry, None).expect("identity fill");
        assert_eq!(rows[0], [0, 1, 2, -1, -1, -1]);
        assert_eq!(counts[0], (0..6u64).rev().collect::<Vec<u64>>());
    }

    /// A profile with the wrong layer count or non-finite weights is refused, not folded into
    /// the identity fill.
    #[test]
    fn a_bad_profile_is_refused() {
        let geometry = BankGeometry::new(2, 4, 2, 3, 8).expect("geometry");
        assert!(fill_from_profile(geometry, Some(&[vec![1.0; 4]])).is_err());
        assert!(
            fill_from_profile(geometry, Some(&[vec![1.0; 4], vec![-1.0, 0.0, 0.0, 0.0]])).is_err()
        );
        assert!(
            fill_from_profile(
                geometry,
                Some(&[vec![1.0; 4], vec![f64::NAN, 0.0, 0.0, 0.0]])
            )
            .is_err()
        );
    }

    /// The stats reader takes the stock layer keys in any object order, refuses a bad key, a
    /// duplicate, an out-of-range index and a missing layer, and returns weights in layer order.
    #[test]
    fn the_usage_dump_round_trips_through_the_reader() {
        let dir = std::env::temp_dir().join(format!("fn-dump-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("dir");
        let path = dir.join("dump.json");
        let geometry = BankGeometry::new(1, 4, 2, 3, 8).expect("geometry");
        let (rows, counts) = fill_from_profile(geometry, None).expect("identity fill");
        let row_refs: Vec<&[i32]> = rows.iter().map(Vec::as_slice).collect();
        {
            let _policy =
                super::ReplaceOnMissLru::from_fill(geometry, &row_refs, &counts, Some(&path))
                    .expect("policy");
        } // drop writes the dump
        assert!(path.exists(), "the drop wrote the dump");
        let read = read_expert_stats(path.to_str().unwrap(), 1).expect("read back");
        assert_eq!(read, vec![vec![0.0; 4]], "zero picks dump zero weights");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn the_stats_reader_parses_and_refuses() {
        let dir = std::env::temp_dir().join(format!("fn-stats-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("dir");
        let write = |name: &str, body: &str| {
            let path = dir.join(name);
            let mut file = std::fs::File::create(&path).expect("file");
            file.write_all(body.as_bytes()).expect("write");
            path
        };
        let good = write(
            "good.json",
            r#"{"model_dir": "x", "layers": {
                "model.language_model.layers.1.mlp": [2.0, 1.0],
                "model.language_model.layers.0.mlp": [0.5, 4.0]}}"#,
        );
        assert_eq!(
            read_expert_stats(good.to_str().unwrap(), 2).expect("read"),
            vec![vec![0.5, 4.0], vec![2.0, 1.0]]
        );
        for (name, body) in [
            ("badkey.json", "{\"layers\": {\"layers.0.mlp\": [1.0]}}"),
            (
                "dup.json",
                r#"{"layers": {"model.language_model.layers.0.mlp": [1.0], "model.language_model.layers.0.mlp": [2.0]}}"#,
            ),
            (
                "range.json",
                "{\"layers\": {\"model.language_model.layers.2.mlp\": [1.0]}}",
            ),
            (
                "missing.json",
                "{\"layers\": {\"model.language_model.layers.0.mlp\": [1.0]}}",
            ),
            ("notjson.json", "nope"),
        ] {
            let path = write(name, body);
            assert!(
                read_expert_stats(path.to_str().unwrap(), 2).is_err(),
                "{name} must be refused"
            );
        }
        std::fs::remove_dir_all(&dir).expect("cleanup");
    }

    /// The served kernels are registered through the shared applier, before the pack is opened
    /// and before any stream or weight exists.
    #[test]
    fn the_served_kernels_are_registered_and_checked_before_the_first_upload() {
        let source = include_str!("start.rs");
        let build = &source[source
            .find("pub(super) fn build_driver(")
            .expect("build_driver")..];
        let call = build
            .find("register_served_kernels(projector.context())?")
            .expect("the registration in build_driver");
        let pack = build
            .find("stage(\"opening the pack\")")
            .expect("the pack open");
        let stream = build.find("ctx.stream()").expect("the stream");
        assert!(
            call < pack && call < stream,
            "register_served_kernels() must precede the pack open and the first stream"
        );
    }
}
