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
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

use super::{
    bank::{
        BankGeometry, ExpertBankState, HostMemory, HostMode, HostPinPlan, HostTerms, RecordSource,
        ReplaceOnMissLru, device::DeviceBank,
    },
    bank_source::PackRecordSource,
    device::CardDevice,
    driver::{DriverShape, TokenDriver},
    executor::{CudaRules, FlashCudaExecutor},
    feeds::StartShape,
    load::{TrunkList, upload::trunk_weights},
    moe_block::{SplitMoe, SplitMoeTrunk},
    pack_feeds::PackFeeds,
    ple_feed::{NgramRows, PleFeed, PleVectors},
    step::{FlashStep, SplitMoeAt},
};
use crate::{
    EngError,
    family::{
        exl3_pack::layer_name,
        flash_next::{
            artifact::tokenizer_pin,
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
        flint::exl3_resident::HostBudget,
        project::cuda::{CudaProjector, CudaWeight, CudaWeights, Reserve, exl3::Exl3Weight},
        qwen36::{cuda::exl3_source::PackReader, tiered::budget::KvPlan},
        session::DecodeSession,
    },
    kernels::cuda::{
        CudaContext, DeviceBuffer,
        resident::exl3::{IndexedOptions, SERVED_SINGLE_ROW},
    },
    sched::worker::{Engine, ReadyNote, Stages, spawn_engine_noted},
    tokenizer::TokenizerBundle,
    weights::exl3::Checkpoint,
};

/// Threads of the host PLE projections (rows split over them).
const PLE_THREADS: usize = 8;

/// The id-indexed options Flash-Next serves its routed experts with: the DP4A decode of codebook 2
/// (the codebook of this pack) and one launch for the 20 gate and up pairs of ten picks. Both write
/// the bytes of the default launches; a set `FIFI_EXL3_DP4A` or `FIFI_EXL3_INDEXED_PAIRS` still wins
/// for its own option.
const SERVED_INDEXED: IndexedOptions = IndexedOptions {
    dp4a: true,
    pairs20: true,
};

/// Registers the kernels the Flash-Next decode runs on its context: the wide single-row kernel for
/// every dense projection and the id-indexed options of the routed experts. Qwen3.6 and Flint
/// register their own; a bad launch switch or an option the kernel cannot combine with is refused
/// here, before any weight is uploaded.
fn register_served_kernels(ctx: &CudaContext) -> Result<(), EngError> {
    ctx.set_single_row_default(SERVED_SINGLE_ROW);
    ctx.set_indexed_default(SERVED_INDEXED);
    ctx.set_odd_rate_kernel(true);
    // The fused-input launch of the dense projections (exl3_decode_fused.h): the input rotation
    // runs inside the projection kernel, two launches per set instead of three, bit-identical
    // bytes (64/64 receipt logits match). Isolated it saves ~1.27 us per launch, but measured on
    // the full 64-step decode it REGRESSES the step: median 39.25 vs 38.28 ms and paired mean
    // +2.25 ms (steps 9-64 of the adjacent off/on pair; REPORT.md cites the artifacts) - the
    // larger shared footprint and the per-column-block rotation recompute reduce the overlap the
    // 253 small launches had with the attention and expert kernels. Kept off for re-measuring
    // after the launch floor changes (e.g. graph capture); `FIFI_FLASH_NEXT_FUSED_IN=1` opts in,
    // any other non-empty value is refused, never a silent fallback. Read once here, per process.
    let fused_in = match std::env::var("FIFI_FLASH_NEXT_FUSED_IN") {
        Err(_) => false,
        Ok(v) if v == "1" => true,
        Ok(v) if v.is_empty() || v == "0" => false,
        Ok(v) => {
            return Err(EngError::BadRequest(format!(
                "FIFI_FLASH_NEXT_FUSED_IN={v}: only 0 and 1 are valid"
            )));
        }
    };
    ctx.set_fused_in_kernel(fused_in);
    ctx.check_single_gather()?;
    eprintln!(
        "flash-next cuda: single-row kernel {}{}",
        ctx.single_row_kernel_name()?,
        ctx.single_gather_suffix()
    );
    if fused_in {
        eprintln!("flash-next cuda: fused-input launch on");
    }
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
) -> Result<(Engine, TokenizerBundle), EngError> {
    let cuda = settings.cuda.as_ref().ok_or_else(|| {
        EngError::BadRequest("engine config: the CUDA decoder is not enabled".to_string())
    })?;
    let manifest = std::fs::read(&settings.manifest).map_err(|error| {
        EngError::Checkpoint(format!("{}: {error}", settings.manifest.display()))
    })?;
    let pin = tokenizer_pin(&manifest, settings.tokenizer_vocab_size)?;
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
    let engine = spawn_engine_noted(prefill_chunk, ready_note, move |stages| {
        make(stages, &artifact, &wanted, rules, guard)
    })?;
    Ok((engine, bundle))
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

fn read_meminfo() -> Result<(u64, u64), EngError> {
    let text = std::fs::read_to_string("/proc/meminfo")
        .map_err(|error| EngError::MemoryPlan(format!("/proc/meminfo: {error}")))?;
    let memory = HostMemory::from_meminfo(&text)?;
    Ok((memory.total, memory.available))
}

/// A record source that adds up the time spent reading and packing records from the pack, so the
/// load log can split the bank fill into the pack read and the pin and upload.
struct TimedSource<'a> {
    inner: &'a PackRecordSource<'a>,
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
}

/// Reads an exllamav3 `expert_stats` JSON into per-layer expert use weights, in layer order.
/// Refuses a missing file, bad JSON, any layer key that is not `model.language_model.layers.<i>.mlp`,
/// a layer index outside `0..layers`, or a duplicate layer. The weights themselves are validated
/// against the bank geometry by [`fill_from_profile`].
/// Greedy water-fill of one uniform slot budget across the layers of a popularity profile:
/// every layer gets the filler minimum first, then each remaining slot goes to the layer whose
/// next hottest still-uncovered expert carries the most profile weight. Layers whose demand
/// fits keep exactly their demand; the total granted never exceeds budget_per_layer * layers.
fn profile_widths(
    profile: &[Vec<f64>],
    budget_per_layer: usize,
    min_slots: usize,
    experts: usize,
) -> Vec<usize> {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    /// A weight ordered totally (profile weights are finite and non-negative, so the bit
    /// pattern preserves their order); the heap needs .
    #[derive(Clone, Copy, PartialEq)]
    struct Weight(f64);
    impl Eq for Weight {}
    impl PartialOrd for Weight {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for Weight {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            self.0.total_cmp(&other.0)
        }
    }
    let layers = profile.len();
    let mut sorted: Vec<Vec<(f64, usize)>> = profile
        .iter()
        .map(|row| {
            let mut by_weight: Vec<(f64, usize)> =
                row.iter().enumerate().map(|(e, &w)| (w, e)).collect();
            by_weight.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
            by_weight
        })
        .collect();
    // (next marginal weight, tie-break layer, cursor) - the heap pops the best next slot.
    // Seeded empty: only the post-drain pushes below enter, all with positive weight.
    let mut heap: BinaryHeap<(Weight, Reverse<usize>, usize)> = BinaryHeap::new();
    let mut widths = vec![0usize; layers];
    let mut remaining = budget_per_layer.saturating_mul(layers);
    // The filler minimum every layer needs, whatever its demand.
    for layer in 0..layers {
        let grant = min_slots.min(sorted[layer].len().min(experts));
        widths[layer] = grant.max(min_slots).min(experts);
        remaining = remaining.saturating_sub(widths[layer]);
    }
    // Cursors start past the granted minimum.
    for layer in 0..layers {
        let cursor = widths[layer].min(sorted[layer].len());
        sorted[layer].drain(..cursor);
        // Only positive demand earns slots: weightless experts never take budget.
        if let Some(&(weight, _)) = sorted[layer].first().filter(|(w, _)| *w > 0.0) {
            heap.push((Weight(weight), Reverse(layer), 0));
        }
    }
    while remaining > 0 {
        let Some((_, Reverse(layer), cursor)) = heap.pop() else {
            break;
        };
        if cursor >= sorted[layer].len() {
            continue;
        }
        widths[layer] += 1;
        remaining -= 1;
        let next = cursor + 1;
        if let Some(&(weight, _)) = sorted[layer].get(next).filter(|(w, _)| *w > 0.0) {
            heap.push((Weight(weight), Reverse(layer), next));
        }
    }
    widths
}

fn read_expert_stats(path: &str, layers: usize) -> Result<Vec<Vec<f64>>, EngError> {
    let bad = |detail: String| {
        EngError::BadRequest(format!("FIFI_FLASH_NEXT_EXPERT_STATS={path}: {detail}"))
    };
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

/// The slot rows and LRU counts of the initial fill. With a profile: per layer, the experts sorted
/// by use weight (highest first, id breaking ties) take the slots in rank order and the weights
/// scale to `u64` counts. Without: the first experts by id take the slots and lower ids count as
/// hotter. Pure; both arms keep every invariant `ReplaceOnMissLru::from_fill` checks.
fn fill_from_profile(
    geometry: BankGeometry,
    profile: Option<&[Vec<f64>]>,
) -> Result<(Vec<Vec<i32>>, Vec<Vec<u64>>), EngError> {
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
                    .map(|e| {
                        if e < geometry.slots_of(layer) {
                            e as i32
                        } else {
                            -1
                        }
                    })
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
        for (slot, &expert) in by_weight.iter().take(geometry.slots_of(layer)).enumerate() {
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
    source: &PackRecordSource<'_>,
) -> Result<DeviceBank, EngError> {
    let (rows, counts) = fill_from_profile(geometry.clone(), profile)?;
    let row_refs: Vec<&[i32]> = rows.iter().map(Vec::as_slice).collect();
    let policy = ReplaceOnMissLru::from_fill(geometry.clone(), &row_refs, &counts)?;

    let state = ExpertBankState::with_policy(geometry, plan.mode, policy)?;
    let timed = TimedSource {
        inner: source,
        spent: Cell::new(Duration::ZERO),
        records: Cell::new(0),
    };
    let opened = Instant::now();
    let bank = DeviceBank::build(projector.context(), state, &timed, plan)?;
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

fn make(
    stages: &Stages<'_>,
    artifact: &Path,
    wanted: &CudaSettings,
    rules: CudaRules,
    guard: BuildGuard,
) -> Result<(FlashCudaExecutor<CardDevice, PackFeeds>, DecodeSession), EngError> {
    let driver = build_driver(
        &|name| stages.begin(name),
        artifact,
        wanted,
        rules.sampler_vocab,
        Some(guard),
    )?;
    FlashCudaExecutor::new(driver, rules)
}

/// Loads the pack and builds the token driver. `guard` pins the one-engine flag at the first leak;
/// a device test passes none.
pub(super) fn build_driver(
    stage: &dyn Fn(&'static str),
    artifact: &Path,
    wanted: &CudaSettings,
    sampler_vocab: usize,
    mut guard: Option<BuildGuard>,
) -> Result<TokenDriver<CardDevice, PackFeeds>, EngError> {
    let began = Instant::now();
    stage("initializing CUDA");
    let projector = CudaProjector::new()?;
    register_served_kernels(projector.context())?;
    let free_vram = projector.memory_info()?.free as u64;

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
    let (mem_total, mem_available) = read_meminfo()?;
    // `FlashLoadPlan::new` replaces the trunk and state bytes (the `(0, 0)`) and the K/V (the
    // placeholder positions) with the numbers the pack and the context give.
    let card = TierInputs::measured(
        tier_shape,
        (0, 0),
        KvPlan {
            positions: 1,
            bytes_per_position: 1,
        },
        free_vram,
        (mem_total, mem_available),
    );
    let load_plan = FlashLoadPlan::new(&report, &checkpoint.model_config, ngram_bytes, card)?;
    let load = load_plan.at_context(
        u64::from(wanted.context),
        wanted.kv_format.format(),
        model.moe.top_k() + 1,
    )?;
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
    // Optional popularity profile (`FIFI_FLASH_NEXT_EXPERT_STATS`, an exllamav3 expert_stats
    // JSON: {"model_dir": .., "layers": {"model.language_model.layers.<i>.mlp": [weights]}}).
    // Unset fills by expert id as before; a set value must read and parse, and a malformed file
    // is refused, never a silent identity fill.
    let profile = match std::env::var("FIFI_FLASH_NEXT_EXPERT_STATS") {
        Err(_) => None,
        Ok(path) if path.is_empty() => None,
        Ok(path) => Some(read_expert_stats(&path, layers)?),
    };
    if profile.is_some() {
        eprintln!(
            "flash-next cuda: expert bank: popularity profile {}",
            std::env::var("FIFI_FLASH_NEXT_EXPERT_STATS").unwrap_or_default()
        );
    }
    // With a profile the same total slot budget moves to where the demand is: every layer
    // starts at the filler minimum, then each remaining slot goes to the layer whose next
    // hottest uncovered expert carries the most weight (a greedy water-fill over the sorted
    // profile rows).
    let widths = match &profile {
        None => vec![slots; layers],
        Some(profile) => profile_widths(profile, slots, model.moe.top_k() + 1, model.moe.experts()),
    };
    let split = widths.iter().filter(|&&w| w != slots).count();
    if split > 0 {
        eprintln!(
            "flash-next cuda: expert bank: profile split {slots} slots/layer: min {}, max {}, {} layers moved",
            widths.iter().min().copied().unwrap_or(slots),
            widths.iter().max().copied().unwrap_or(slots),
            split
        );
    }
    eprintln!("{load}; serving {slots} slots per layer");
    if sampler_vocab == 0 || sampler_vocab > model.vocab {
        return Err(EngError::Checkpoint(format!(
            "sampler vocabulary {} is outside 1..={}, the width of the model head",
            sampler_vocab, model.vocab
        )));
    }

    // The host side is admitted before anything is uploaded: the pinned store of the bank, the
    // engine heap and margin of `HostTerms`, and the two PLE projection matrices that
    // `PleVectors::read` holds in F32 (`HostTerms::engine_heap` does not count them).
    let record_bytes = PackRecordSource::new(&checkpoint)?.layout().record_bytes();
    let geometry = BankGeometry::new_widths(
        layers,
        model.moe.experts(),
        model.moe.top_k(),
        &widths,
        record_bytes,
    )?;
    // The store mode: an inclusive store holds every expert pinned, so a miss copies one way
    // only (the D2H write-back of the evicted expert disappears); it needs every record's bytes
    // of host RAM (43.07 GiB on this pack), so `auto` takes it only when the plan admits it and
    // falls back to the exclusive store of spare places. `FIFI_FLASH_NEXT_BANK_MODE` forces a
    // mode (`inclusive` or `exclusive`); anything else is refused, never a silent fallback.
    let memory = HostMemory {
        total: mem_total,
        available: mem_available,
    };
    let plan_for = |geometry: &BankGeometry| -> Result<HostPinPlan, EngError> {
        match std::env::var("FIFI_FLASH_NEXT_BANK_MODE").as_deref() {
            Err(_) | Ok("") | Ok("auto") => HostPinPlan::new(
                geometry,
                HostMode::Inclusive,
                &memory,
                &HostTerms::DEFAULT,
            )
            .or_else(|inclusive| {
                eprintln!(
                    "flash-next cuda: inclusive store refused ({inclusive:?}); exclusive store"
                );
                HostPinPlan::new(geometry, HostMode::Exclusive, &memory, &HostTerms::DEFAULT)
            }),
            Ok("inclusive") => {
                HostPinPlan::new(geometry, HostMode::Inclusive, &memory, &HostTerms::DEFAULT)
            }
            Ok("exclusive") => {
                HostPinPlan::new(geometry, HostMode::Exclusive, &memory, &HostTerms::DEFAULT)
            }
            Ok(other) => Err(EngError::BadRequest(format!(
                "FIFI_FLASH_NEXT_BANK_MODE={other}: only auto, inclusive and exclusive are valid"
            ))),
        }
    };
    let mut geometry = geometry;
    let mut host_plan = plan_for(&geometry);
    // A profile split the widths and no store mode admits it (the inclusive store needs the
    // whole pack pinned and the exclusive place budget is uniform): keep serving on the uniform
    // widths - the profile still orders the fill, only the split is dropped. A forced mode
    // keeps refusing instead.
    if host_plan.is_err()
        && geometry.uniform_slots().is_none()
        && !matches!(
            std::env::var("FIFI_FLASH_NEXT_BANK_MODE").as_deref(),
            Ok("inclusive") | Ok("exclusive")
        )
    {
        eprintln!(
            "flash-next cuda: no store admits the profile slot split; keeping uniform {slots} slots per layer (the profile still orders the fill)"
        );
        let uniform = vec![slots; layers];
        geometry = BankGeometry::new_widths(
            layers,
            model.moe.experts(),
            model.moe.top_k(),
            &uniform,
            record_bytes,
        )?;
        host_plan = plan_for(&geometry);
    }
    let host_plan = host_plan?;
    eprintln!(
        "flash-next cuda: {} store, {} pinned host bytes",
        match host_plan.mode {
            HostMode::Inclusive => "inclusive",
            HostMode::Exclusive => "exclusive",
        },
        host_plan.store_bytes
    );
    let ple_width = (model.hyper.hidden() * model.hyper.branches() + model.hyper.hidden()) as u64;
    let ple_bytes = ple_width * model.ple.embed() as u64 * 4;
    HostBudget::probe()?.admit_for(
        host_plan.process_bytes + ple_bytes,
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
    let source = PackRecordSource::new(checkpoint)?;
    let bank = build_bank(projector, geometry, &host_plan, profile.as_deref(), &source)?;
    eprintln!(
        "flash-next cuda: expert bank filled at {:.1} s",
        began.elapsed().as_secs_f64()
    );

    stage("building the step");
    let stream = Box::leak(Box::new(ctx.stream()?));
    let step = FlashStep::new(
        ctx,
        stream,
        weights,
        &checkpoint.model_config,
        wanted.context as usize,
        0..layers,
    )?;
    let block = SplitMoe::new(
        ctx,
        moe_trunks(weights, layers)?,
        bank,
        source.layout(),
        &step.moe_dims(),
    )?;
    let model_layers: Vec<usize> = (0..layers).collect();
    let moe = SplitMoeAt::new(block, &model_layers)?;
    let device = CardDevice::new(step, moe);

    stage("opening the n-gram table");
    let (vectors, projection) =
        PleVectors::read(&checkpoint.source, model.ple_layer, &model.ple, PLE_THREADS)?;
    let ple = PleFeed::new(
        model.ple,
        NgramRows::new(ngram)?,
        vectors,
        Box::new(projection),
    );
    let hidden = model.hyper.hidden();
    let branches = model.hyper.branches();
    let ple_embed = model.ple.embed();
    let feeds = PackFeeds::new(&checkpoint.source, ple, hidden, branches);
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
    let driver = TokenDriver::new(device, feeds, shape)?;
    eprintln!(
        "flash-next cuda: ready in {:.1} s",
        began.elapsed().as_secs_f64()
    );
    Ok(driver)
}

#[cfg(test)]
mod tests {
    use super::{BankGeometry, HostMode, fill_from_profile, profile_widths, read_expert_stats};
    use std::io::Write as _;

    /// The water-fill keeps the budget, gives every layer the filler minimum, spends the rest
    /// on marginal weight, and never exceeds a layer's demand or the expert count; uniform
    /// demand degenerates to the uniform split.
    #[test]
    fn the_water_fill_moves_budget_to_demand() {
        let profile = vec![vec![9.0, 9.0, 0.0, 0.0], vec![1.0, 1.0, 1.0, 1.0]];
        let widths = profile_widths(&profile, 3, 2, 4);
        assert_eq!(widths.iter().sum::<usize>(), 3 * 2, "the budget is kept");
        assert!(widths.iter().all(|&w| w >= 2), "the filler minimum");
        assert_eq!(
            widths,
            vec![2, 4],
            "layer 0 has demand 2, layer 1 takes the rest"
        );
        let widths = profile_widths(&profile, 2, 2, 4);
        assert_eq!(widths, vec![2, 2], "a tight budget is the uniform minimum");
        // Capped demand: zero-weight experts never earn a slot beyond the minimum.
        let profile = vec![vec![5.0, 0.0, 0.0, 0.0], vec![0.0; 4]];
        let widths = profile_widths(&profile, 4, 2, 4);
        assert_eq!(
            widths,
            vec![2, 2],
            "no layer exceeds its demand or gains weightless slots"
        );
    }

    /// A per-layer bank validates each width like the uniform one and reports its widths; the
    /// exclusive host mode refuses it, the inclusive one admits it.
    #[test]
    fn a_bank_with_per_layer_widths_validates_and_splits_places() {
        assert!(BankGeometry::new_widths(2, 16, 2, &[3, 9], 8).is_ok());
        assert!(
            BankGeometry::new_widths(2, 8, 2, &[3, 9], 8).is_err(),
            "a width past the experts"
        );
        assert!(
            BankGeometry::new_widths(2, 16, 2, &[3], 8).is_err(),
            "a width per layer"
        );
        assert!(
            BankGeometry::new_widths(2, 16, 2, &[2, 9], 8).is_err(),
            "below the minimum"
        );
        assert!(
            BankGeometry::new_widths(2, 16, 2, &[3, 9], 7).is_err(),
            "not whole floats"
        );
        let geometry = BankGeometry::new_widths(2, 16, 2, &[3, 9], 8).expect("geometry");
        assert_eq!(geometry.slots_of(0), 3);
        assert_eq!(geometry.slots_of(1), 9);
        assert_eq!(geometry.uniform_slots(), None);
        assert_eq!(geometry.device_bytes(0), 3 * 8);
        assert_eq!(geometry.device_bytes(1), 9 * 8);
        assert_eq!(geometry.device_bytes_total(), (3 + 9) * 8);
        let places = HostMode::Exclusive.places_per_layer(&geometry);
        assert!(places.is_err(), "the exclusive place budget is uniform");
        assert_eq!(
            HostMode::Inclusive
                .places_per_layer(&geometry)
                .expect("places"),
            16
        );
    }

    /// A profile fills the slots in weight rank order (ties by id) and scales the weights to
    /// counts; the same bank without one keeps the id-ordered fill and the reversed-id counts.
    #[test]
    fn a_profile_ranks_the_initial_fill() {
        let geometry = BankGeometry::new(1, 6, 2, 3, 8).expect("geometry");
        let profile = [vec![1.0, 9.0, 4.0, 9.0, 0.0, 2.0]];
        let (rows, counts) = fill_from_profile(geometry.clone(), Some(&profile)).expect("fill");
        // experts 1 and 3 tie at the top weight; id breaks it, so 1 takes slot 0, 3 slot 1, 2 slot 2.
        assert_eq!(rows[0], [-1, 0, 2, 1, -1, -1]);
        assert_eq!(counts[0], vec![1000, 9000, 4000, 9000, 0, 2000]);
        let (rows, counts) = fill_from_profile(geometry.clone(), None).expect("identity fill");
        assert_eq!(rows[0], [0, 1, 2, -1, -1, -1]);
        assert_eq!(counts[0], (0..6u64).rev().collect::<Vec<u64>>());
    }

    /// A profile with the wrong layer count or non-finite weights is refused, not folded into
    /// the identity fill.
    #[test]
    fn a_bad_profile_is_refused() {
        let geometry = BankGeometry::new(2, 4, 2, 3, 8).expect("geometry");
        assert!(fill_from_profile(geometry.clone(), Some(&[vec![1.0; 4]])).is_err());
        assert!(
            fill_from_profile(
                geometry.clone(),
                Some(&[vec![1.0; 4], vec![-1.0, 0.0, 0.0, 0.0]])
            )
            .is_err()
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

    /// The served kernels are registered and checked before the pack is opened and before any
    /// stream or weight exists, in the order set default, set options, check.
    #[test]
    fn the_served_kernels_are_registered_and_checked_before_the_first_upload() {
        let source = include_str!("start.rs");
        let register = &source[source
            .find("fn register_served_kernels(")
            .expect("register_served_kernels")..];
        let single = register
            .find("ctx.set_single_row_default(")
            .expect("the single-row default");
        let indexed = register
            .find("ctx.set_indexed_default(")
            .expect("the indexed default");
        let odd = register
            .find("ctx.set_odd_rate_kernel(true)")
            .expect("the odd-rate kernel switch");
        let check = register
            .find("ctx.check_single_gather()?")
            .expect("the switch check");
        assert!(
            single < indexed && indexed < odd && odd < check,
            "the defaults must be registered before check_single_gather()"
        );
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
