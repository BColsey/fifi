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
    ctx.check_single_gather()?;
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

/// The bank: the first `slots` experts of every layer on the card, the rest pinned on the host,
/// under the host plan the caller admitted.
fn build_bank(
    projector: &CudaProjector,
    geometry: BankGeometry,
    plan: &HostPinPlan,
    source: &PackRecordSource<'_>,
) -> Result<DeviceBank, EngError> {
    let rows: Vec<Vec<i32>> = (0..geometry.layers)
        .map(|_| {
            (0..geometry.experts)
                .map(|expert| {
                    if expert < geometry.slots {
                        expert as i32
                    } else {
                        -1
                    }
                })
                .collect()
        })
        .collect();
    let row_refs: Vec<&[i32]> = rows.iter().map(Vec::as_slice).collect();
    // No popularity profile: lower ids count as hotter, so the first experts are the last evicted.
    let counts = vec![(0..geometry.experts as u64).rev().collect::<Vec<u64>>(); geometry.layers];
    let policy = ReplaceOnMissLru::from_fill(geometry, &row_refs, &counts)?;
    let state = ExpertBankState::with_policy(geometry, HostMode::Exclusive, policy)?;
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
    let geometry = BankGeometry::new(
        layers,
        model.moe.experts(),
        model.moe.top_k(),
        slots,
        record_bytes,
    )?;
    let host_plan = HostPinPlan::new(
        &geometry,
        HostMode::Exclusive,
        &HostMemory {
            total: mem_total,
            available: mem_available,
        },
        &HostTerms::DEFAULT,
    )?;
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
    let bank = build_bank(projector, geometry, &host_plan, &source)?;
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
        let check = register
            .find("ctx.check_single_gather()?")
            .expect("the switch check");
        assert!(
            single < indexed && indexed < check,
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
