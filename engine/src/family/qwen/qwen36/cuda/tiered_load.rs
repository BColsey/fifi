//! The load path of the tiered step: which tensors a tiered load uploads, what the bank sizes are
//! checked against, and the host side built over the mapped GGUF.
//!
//! WHAT: [`names_without_experts`] drops the routed expert stacks from the device tensor list (the
//! banks hold the hot experts instead), [`admit_banks`] refuses a bank that does not fit the free
//! device memory through the admission arithmetic of `tiered/budget.rs`, and [`build_tiered`]
//! allocates the banks and builds the [`TieredHost`] over a read-only mapping of the checkpoint.
//! WHY: the 20 GB checkpoint cannot be resident in 6 GiB; everything but the experts still is.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use super::super::settings::SlotRequest;
use super::super::tiered::cpu_pool::{CpuPool, ExpertKernel, MappedRegion, PoolError};
use super::super::tiered::guess::{GuessError, LayerRouter, RouterGuess};
use super::super::tiered::host::TieredHost;
use super::super::tiered::locator::{ExpertLocator, ExpertTensorNames};
use super::super::tiered::{PolicyMode, TieredConfig};
use super::dims::{StepDims, TieredPlan};
use super::step::QwenResidentStep;
use super::tiered_step::{ColdSet, ColdStaging};
use crate::EngError;
use crate::family::device_budget::KvPlan;
use crate::family::expert_bank::pinned_region::PinnedRegion;
use crate::family::mapped_file::MappedFile;
use crate::family::mem::HostFacts;
use crate::family::tier_budget::{ExpertSize, VramBudget, plan_vram};
use crate::kernels::cuda::{CudaContext, ExpertBank, ExpertBankGeometry};
use crate::kernels::expert_cpu::ExpertShape;
use crate::kernels::expert_cpu_q8k::DotBackend;
use crate::weights::{GgufExpertIndex, GgufHeader, GgufType, LocatorError};

/// The expert kernel of the tier's CPU pool: the per-32 group grid of the device path, not llama.cpp's
/// Q8_K. Gate (b) failed with Q8_K at 80 percent CPU-served picks (p95 0.686 against the bound 0.651) and
/// passed with this grid (0.478). The choice is pinned by the pure test below and, on the device, by the
/// rule-C bounds; [`tier_pool`] is where the load and those tests share it.
fn tier_kernel() -> ExpertKernel {
    ExpertKernel::Q8g(DotBackend::detect())
}

/// The CPU pool of the tier: `threads` workers running [`tier_kernel`] over experts of `shape`. The load and
/// the device rule-C tests both build their pool here, and a source pin keeps `build_tiered` on this call.
pub(super) fn tier_pool(threads: usize, shape: ExpertShape) -> Result<CpuPool, PoolError> {
    CpuPool::with_kernel(threads, shape, tier_kernel())
}

/// The three routed expert stacks of a layer, by the suffix of their GGUF names.
const EXPERT_SUFFIXES: [&str; 3] = [
    ".ffn_gate_exps.weight",
    ".ffn_up_exps.weight",
    ".ffn_down_exps.weight",
];

/// The device tensor names of a tiered load: every name but the routed expert stacks.
pub(super) fn names_without_experts<'n>(names: &[&'n str]) -> Vec<&'n str> {
    names
        .iter()
        .copied()
        .filter(|name| !EXPERT_SUFFIXES.iter().any(|suffix| name.ends_with(suffix)))
        .collect()
}

/// The bank geometry of one layer whose stacks the file stores in `quant`.
pub(super) fn geometry_of(dims: &StepDims, quant: [GgufType; 3]) -> ExpertBankGeometry {
    ExpertBankGeometry {
        hidden: dims.shape.hidden,
        inter: dims.shape.moe.inter,
        top_k: dims.shape.moe.top_k,
        quant,
    }
}

/// The record types of the gate, up and down stacks of every layer, read from the header through
/// the shared expert index; a type without a native kernel is refused with the tensor named.
pub(super) fn expert_quant(
    header: &GgufHeader,
    dims: &StepDims,
) -> Result<Vec<[GgufType; 3]>, EngError> {
    let (layers, moe) = (dims.shape.kinds.len(), &dims.shape.moe);
    let shape = (dims.shape.hidden, moe.inter);
    let refuse = |error: LocatorError| EngError::Checkpoint(format!("tiered: {error:?}"));
    let index = GgufExpertIndex::new(
        header,
        &[header.file_len()],
        &ExpertTensorNames::standard(),
        layers,
        moe.experts,
        shape,
        None,
    )
    .map_err(refuse)?;
    index
        .require_native_rows(shape.0, shape.1)
        .map_err(refuse)?;
    (0..layers)
        .map(|layer| index.three_types(layer).map_err(refuse))
        .collect()
}

/// The geometry that sizes the plan: the layer whose three records are largest, so the bank and
/// staging budgets hold for every layer of a file that mixes types.
pub(super) fn planning_geometry(
    dims: &StepDims,
    quants: &[[GgufType; 3]],
) -> Result<ExpertBankGeometry, EngError> {
    ExpertBankGeometry::widest(geometry_of(dims, [GgufType::Q4K; 3]), quants)
}

/// The device bytes of one expert record in each layer: a file that mixes record types has a
/// different size per layer.
pub(super) fn layer_slot_bytes(
    dims: &StepDims,
    quants: &[[GgufType; 3]],
) -> Result<Vec<u64>, EngError> {
    quants
        .iter()
        .map(|quant| {
            let geometry = geometry_of(dims, *quant);
            Ok(ExpertBank::bytes_for(geometry, geometry.top_k)? / geometry.top_k as u64)
        })
        .collect()
}

/// The device bytes of one expert slot in one layer's bank.
pub(super) fn bank_slot_bytes(geometry: ExpertBankGeometry) -> Result<u64, EngError> {
    let minimum_bank = ExpertBank::bytes_for(geometry, geometry.top_k)?;
    Ok(minimum_bank / geometry.top_k as u64)
}

/// Refuses a plan whose banks do not fit `free` device bytes (after the weights, the KV cache
/// and the step state are resident), through `plan_vram`: the minimum bank is the plan's own
/// slots per layer, the desktop reserve is the loader's headroom, and `cold_bytes` (the chunk
/// prefill's staging bank, when a nonzero one is derived) is charged as already spoken for
/// inside `usable`. The message names the numbers.
pub(super) fn admit_banks(
    free: u64,
    geometry: ExpertBankGeometry,
    layers: usize,
    request: SlotRequest,
    experts: usize,
    reserve: u64,
    cold_bytes: u64,
) -> Result<usize, EngError> {
    let minimum = match request {
        SlotRequest::Fit => geometry.top_k,
        SlotRequest::Exactly(count) => count,
    };
    if minimum < geometry.top_k || minimum > experts {
        return Err(EngError::MemoryPlan(format!(
            "tiered: {minimum} slots per layer is outside the model range {}..={experts}",
            geometry.top_k
        )));
    }
    let bytes_per_slot = bank_slot_bytes(geometry)?;
    let budget = VramBudget {
        usable: free,
        non_expert: cold_bytes,
        state: 0,
        arena: 0,
        desktop_reserve: reserve,
    };
    // The KV cache is already allocated by the step: it takes no part in this split.
    let kv = KvPlan {
        positions: 0,
        bytes_per_position: 1,
    };
    let expert = ExpertSize {
        bytes: bytes_per_slot,
        layers,
    };
    let capacity = plan_vram(&budget, &kv, &expert, minimum, 0).map_err(|error| {
        EngError::MemoryPlan(format!(
            "tiered: {minimum} slots per layer of {layers} layers do not fit {free} free device bytes with {reserve} reserved: {error:?}"
        ))
    })?;
    Ok(match request {
        SlotRequest::Fit => capacity.bank_slots_per_layer.min(experts),
        SlotRequest::Exactly(count) => count,
    })
}

/// The cold staging depth the free device memory funds: the hottest distinct cold experts
/// per layer top out at 78 on this model, and every cap below that pushes experts back
/// into the 0.24 ms/pair CPU pool, so the target is 78 whenever the spare above the
/// reserve (after every layer's banks) reaches it. A probe-ceiling config leaves less
/// than one bank-slot-set of spare by construction (`plan_vram` floors the slots), so the
/// formula yields 0 there and the operator funds the staging by dropping 2 slots per
/// layer (two slot-sets free 141,557,760 B >= the 138,018,816 B of 78 experts).
pub(super) fn cold_cap_for(
    free: u64,
    geometry: ExpertBankGeometry,
    (layers, stagings): (usize, usize),
    slots_per_layer: usize,
    reserve: u64,
) -> usize {
    /// The measured maximum of DISTINCT cold experts serving one layer's picks.
    const COLD_TARGET: usize = super::prefill::COLD_CHANNELS_MAX;
    let per_expert = ExpertBank::bytes_for(geometry, geometry.top_k).unwrap_or(u64::MAX)
        / u64::try_from(geometry.top_k).unwrap_or(1);
    let banks = ExpertBank::bytes_for(geometry, slots_per_layer)
        .unwrap_or(u64::MAX)
        .saturating_mul(u64::try_from(layers).unwrap_or(u64::MAX));
    let spare = free.saturating_sub(reserve).saturating_sub(banks);
    // Every distinct type triple holds a staging bank of the same depth.
    let per_depth = per_expert.saturating_mul(u64::try_from(stagings.max(1)).unwrap_or(1));
    COLD_TARGET.min(usize::try_from(spare / per_depth.max(1)).unwrap_or(0))
}

/// Whether the tiered session takes the chunked prefill arm: always, unless `Q36_TIERED_CHUNK=0`.
/// The arm reads every routed and shared expert in the type its file stores, so no file turns it
/// off.
pub(super) fn tiered_chunk_arm() -> bool {
    super::prefill::tiered_chunk_enabled(
        std::env::var_os(super::prefill::TIERED_CHUNK_VAR).as_deref(),
    )
}

pub(super) fn build_tiered(
    ctx: &CudaContext,
    (header, gguf): (&GgufHeader, &Path),
    dims: &StepDims,
    plan: &TieredPlan,
    slots: usize,
    cold_cap: usize,
) -> Result<(Vec<ExpertBank>, TieredHost, ColdSet), EngError> {
    let moe = &dims.shape.moe;
    let layers = dims.shape.kinds.len();
    let shape = ExpertShape {
        hidden: dims.shape.hidden,
        inter: moe.inter,
    };
    let checkpoint = |what: String| EngError::Checkpoint(format!("tiered: {what}"));
    let locator = ExpertLocator::new(
        header,
        &ExpertTensorNames::standard(),
        layers,
        moe.experts,
        shape,
    )
    .map_err(|error| checkpoint(format!("{error:?}")))?;
    let mapped = MappedFile::open(gguf, header.file_len())
        .map_err(|error| checkpoint(format!("mapping {}: {error}", gguf.display())))?;
    let region: Arc<dyn MappedRegion> = match pinned_copy(&mapped) {
        Some(pinned) => Arc::new(pinned),
        None => Arc::new(mapped),
    };
    let pool = tier_pool(plan.threads, shape)
        .map_err(|error| checkpoint(format!("cpu pool: {error:?}")))?;
    let config = TieredConfig {
        layers,
        experts: moe.experts,
        top_k: moe.top_k,
        bank_slots_per_layer: slots,
        mode: PolicyMode::Adaptive,
        swaps_per_token: f64::from(plan.swaps_milli) / 1000.0,
        decay: f64::from(plan.decay_milli) / 1000.0,
    };
    let timeout = Duration::from_millis(u64::from(plan.timeout_ms));
    // The guess reads the same mapping the host will own, so build it first. Its advice only
    // decides which experts the banks prefetch, so a checkpoint whose routers it cannot read
    // (anything but plain F32) serves without it.
    let guess = match build_router_guess(header, &region, dims) {
        Ok(guess) => Some(guess),
        Err(error) => {
            eprintln!("q36 cuda: router-guess prefetch off: {error:?}");
            None
        }
    };
    let mut host = TieredHost::new(config, locator, region, pool, timeout)
        .map_err(|error| checkpoint(error.to_string()))?;
    if let Some(guess) = guess {
        host.set_guess(guess, moe.top_k);
    }
    let quants = expert_quant(header, dims)?;
    let geometry = planning_geometry(dims, &quants)?;
    let banks = quants
        .iter()
        .map(|quant| ExpertBank::new(ctx, geometry_of(dims, *quant), slots))
        .collect::<Result<Vec<_>, _>>()?;
    // The staging banks follow the banks: same free pool, one per distinct record-type triple (a
    // staging bank is typed like the stacks it serves). A cap below top_k cannot even build a
    // bank geometry, so it builds none.
    let cold = if cold_cap >= geometry.top_k {
        let mut stagings = Vec::new();
        let mut seen: Vec<[GgufType; 3]> = Vec::new();
        let mut of_layer = Vec::with_capacity(quants.len());
        for quant in &quants {
            let index = match seen.iter().position(|known| known == quant) {
                Some(index) => index,
                None => {
                    stagings.push(ColdStaging::new(ctx, geometry_of(dims, *quant), cold_cap)?);
                    seen.push(*quant);
                    stagings.len() - 1
                }
            };
            of_layer.push(Some(index));
        }
        ColdSet::new(stagings, of_layer)
    } else {
        ColdSet::none()
    };
    Ok((banks, host, cold))
}

/// Admits the banks against the free device memory (the weights, the KV cache and the step state
/// are resident), builds the banks and the host and attaches them to `step`: the one tail shared by
/// the serve start and the gate. `source` is the checkpoint header and its path.
pub(super) fn attach_tiered_parts(
    step: &mut QwenResidentStep<'_>,
    ctx: &CudaContext,
    (source, dims): ((&GgufHeader, &Path), &StepDims),
    plan: &TieredPlan,
    (sessions, growth_reserve): (usize, u64),
) -> Result<usize, EngError> {
    let free = u64::try_from(ctx.memory_info()?.free).unwrap_or(u64::MAX);
    // The banks leave the loader's headroom and the parts of the sessions beside them, by the
    // same rule the card-short planner used to size the slots.
    let sessions_reserve = super::super::card_short::tiered_session_reserve(
        step.seq_bytes_at(
            step.seq_cells()
                .min(crate::family::cell_pool::INITIAL_CELLS),
        ),
        sessions,
    )
    .ok_or_else(|| EngError::MemoryPlan("tiered: the sessions' reserve overflows".into()))?;
    let reserve = ADMISSION_RESERVE
        .saturating_add(sessions_reserve)
        .saturating_add(growth_reserve);
    let layers = dims.shape.kinds.len();
    let quants = expert_quant(source.0, dims)?;
    let geometry = planning_geometry(dims, &quants)?;
    // One staging bank per distinct record-type triple (see `build_tiered`).
    let mut distinct: Vec<[GgufType; 3]> = Vec::new();
    for quant in &quants {
        if !distinct.contains(quant) {
            distinct.push(*quant);
        }
    }
    let planned = match plan.slots {
        SlotRequest::Fit => geometry.top_k,
        SlotRequest::Exactly(count) => count,
    };
    let cold_cap = cold_cap_for(
        free,
        geometry,
        (layers, distinct.len()),
        planned,
        reserve,
    );
    // Charge the staging the plan is about to allocate, so the reserve the banks leave
    // behind still covers the loader's headroom after the staging exists.
    let mut cold_bytes = 0u64;
    if cold_cap >= geometry.top_k {
        for quant in &distinct {
            cold_bytes += ExpertBank::bytes_for(geometry_of(dims, *quant), cold_cap)?;
        }
    }
    let slots = admit_banks(
        free,
        geometry,
        layers,
        plan.slots,
        dims.shape.moe.experts,
        reserve,
        cold_bytes,
    )?;
    let (banks, host, cold) = build_tiered(ctx, source, dims, plan, slots, cold_cap)?;
    step.attach_tiered(banks, host, cold)?;
    if plan.preload {
        step.preload_tiered()?;
    }
    Ok(slots)
}

/// A page-locked copy of the checkpoint for the CPU pool and the bank fills to read, when the
/// host budget admits one; `None` keeps them on the mapping. The line says which one serves.
fn pinned_copy(mapped: &MappedFile) -> Option<PinnedRegion> {
    let label = "q36 cuda: tiered expert store";
    let facts = match HostFacts::read(label, label) {
        Ok(facts) => facts,
        Err(error) => {
            eprintln!(
                "{label}: the host memory is unreadable ({error:?}); experts stay in the page cache"
            );
            return None;
        }
    };
    match PinnedRegion::copy_of(mapped.as_ref(), &facts, label) {
        Ok(pinned) => {
            eprintln!(
                "{label}: {} bytes held in page-locked memory",
                mapped.as_ref().len()
            );
            Some(pinned)
        }
        Err(error) => {
            eprintln!("{label}: experts stay in the page cache ({error:?})");
            None
        }
    }
}

/// The device bytes kept free beside the banks: the loader's headroom (1 GiB).
const ADMISSION_RESERVE: u64 = crate::family::mem::DEVICE.load_headroom;

/// Builds the [`RouterGuess`] of the whole model straight from the checkpoint mapping:
/// every layer's `ffn_gate_inp` router matrix and `post_attention_layernorm` weight are
/// plain F32 in this family's files, so no dequantization is involved. `Err` names the
/// first tensor that is missing or not the expected F32 `[experts, hidden]` shape.
fn build_router_guess(
    header: &GgufHeader,
    region: &Arc<dyn MappedRegion>,
    dims: &StepDims,
) -> Result<RouterGuess, GuessError> {
    let (hidden, experts) = (dims.shape.hidden, dims.shape.moe.experts);
    let eps = crate::family::spec::QWEN3_6_35B_A3B.rms_norm_eps;
    let bytes = region.bytes();
    let read_f32 = |tensor: &crate::weights::GgufTensor, len: usize, what: &str| {
        let range = tensor.abs_range();
        let start = usize::try_from(range.start).map_err(|_| {
            GuessError::BadInput(format!(
                "tiered guess: {what} offset {} overflows usize",
                range.start
            ))
        })?;
        let end = start
            .checked_add(tensor.byte_len() as usize)
            .ok_or_else(|| {
                GuessError::BadInput(format!("tiered guess: {what} length overflows usize"))
            })?;
        let slice = bytes.get(start..end).ok_or_else(|| {
            GuessError::BadInput(format!(
                "tiered guess: {what} range {start}..{end} is outside the file"
            ))
        })?;
        if tensor.ggml_type() != crate::weights::GgufType::F32
            || tensor.element_count() != len as u64
        {
            return Err(GuessError::BadInput(format!(
                "tiered guess: {what} is not F32 of {len} values"
            )));
        }
        let (words, _) = slice.as_chunks::<4>();
        Ok(words
            .iter()
            .map(|word| f32::from_le_bytes(*word))
            .collect::<Vec<f32>>()
            .into())
    };
    let mut routers = Vec::with_capacity(dims.shape.kinds.len());
    for layer in 0..dims.shape.kinds.len() {
        let router = header
            .tensor(&format!("blk.{layer}.ffn_gate_inp.weight"))
            .ok_or_else(|| {
                GuessError::BadInput(format!(
                    "tiered guess: blk.{layer}.ffn_gate_inp.weight is missing"
                ))
            })?;
        let norm = header
            .tensor(&format!("blk.{layer}.post_attention_norm.weight"))
            .ok_or_else(|| {
                GuessError::BadInput(format!(
                    "tiered guess: blk.{layer}.post_attention_norm.weight is missing"
                ))
            })?;
        routers.push(LayerRouter {
            router: read_f32(router, experts * hidden, "the router matrix")?,
            norm_weight: read_f32(norm, hidden, "the post-attention norm weight")?,
        });
    }
    RouterGuess::new(routers, hidden, experts, eps)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::qwen::qwen36::cuda::common_tests::dims;

    #[test]
    fn the_tier_kernel_is_the_per_32_group_grid_never_q8_k() {
        assert!(
            matches!(tier_kernel(), ExpertKernel::Q8g(_)),
            "{:?}",
            tier_kernel()
        );
        let shape = ExpertShape {
            hidden: 256,
            inter: 256,
        };
        let pool = tier_pool(2, shape).expect("a pool");
        assert!(
            matches!(pool.kernel(), ExpertKernel::Q8g(_)),
            "{:?}",
            pool.kernel()
        );
        assert_eq!(pool.threads(), 2);
    }

    #[test]
    fn a_tiered_load_drops_exactly_the_routed_expert_stacks() {
        let names = [
            "token_embd.weight",
            "blk.0.ffn_gate_inp.weight",
            "blk.0.ffn_gate_exps.weight",
            "blk.0.ffn_up_exps.weight",
            "blk.0.ffn_down_exps.weight",
            "blk.0.ffn_gate_shexp.weight",
            "blk.39.ffn_down_exps.weight",
            "output.weight",
        ];
        assert_eq!(
            names_without_experts(&names),
            [
                "token_embd.weight",
                "blk.0.ffn_gate_inp.weight",
                "blk.0.ffn_gate_shexp.weight",
                "output.weight"
            ]
        );
    }

    #[test]
    fn the_bank_geometry_comes_from_the_step_dims() {
        let dims = dims();
        let geometry = geometry_of(&dims, [GgufType::Q4K; 3]);
        assert_eq!(geometry.hidden, dims.shape.hidden);
        assert_eq!(geometry.inter, dims.shape.moe.inter);
        assert_eq!(geometry.top_k, dims.shape.moe.top_k);
    }

    #[test]
    fn auto_slots_fit_whole_layer_sets_above_the_reserve_up_to_the_model_limit() {
        for geometry in [
            ExpertBankGeometry::q4k(2048, 512, 8),
            ExpertBankGeometry::q4k(1024, 256, 4),
        ] {
            for layers in [12, 40] {
                let minimum = geometry.top_k;
                let experts = 32;
                let reserve = 1 << 30;
                let slot_set = ExpertBank::bytes_for(geometry, minimum).unwrap() / minimum as u64
                    * layers as u64;
                let fit = |free| {
                    admit_banks(
                        free,
                        geometry,
                        layers,
                        SlotRequest::Fit,
                        experts,
                        reserve,
                        0,
                    )
                };
                assert!(fit(reserve + slot_set * minimum as u64 - 1).is_err());
                for (sets, extra, want) in [
                    (minimum, 0, minimum),
                    (18, slot_set - 1, 18),
                    (19, 0, 19),
                    (64, 0, experts),
                ] {
                    let free = reserve + slot_set * sets as u64 + extra;
                    assert_eq!(fit(free).unwrap(), want);
                    assert!(
                        ExpertBank::bytes_for(geometry, want).unwrap() * layers as u64
                            <= free - reserve
                    );
                }
                assert_eq!(
                    admit_banks(
                        reserve + 28 * slot_set,
                        geometry,
                        layers,
                        SlotRequest::Exactly(16),
                        experts,
                        reserve,
                        0
                    )
                    .unwrap(),
                    16
                );
                assert!(
                    admit_banks(
                        u64::MAX,
                        geometry,
                        layers,
                        SlotRequest::Exactly(experts + 1),
                        experts,
                        reserve,
                        0
                    )
                    .is_err()
                );
            }
        }
    }

    #[test]
    fn a_plan_that_does_not_fit_is_refused_naming_the_budget() {
        let geometry = geometry_of(&dims(), [GgufType::Q4K; 3]);
        let top_k = geometry.top_k as u64;
        let slot_set = ExpertBank::bytes_for(geometry, geometry.top_k).unwrap() / top_k * 40;
        let plan = SlotRequest::Exactly;
        let reserve = 1 << 30;
        // 40 slots per layer need 40 sets; a card with room for 39 plus the reserve refuses.
        let free = reserve + 39 * slot_set;
        let refusal = admit_banks(free, geometry, 40, plan(40), 256, reserve, 0);
        match refusal {
            Err(EngError::MemoryPlan(message)) => {
                assert!(message.contains("40 slots per layer"), "{message}");
                assert!(message.contains(&free.to_string()), "{message}");
                assert!(message.contains("minimum expert bank"), "{message}");
            }
            other => panic!("39 slots of room must refuse 40: {other:?}"),
        }
        assert!(admit_banks(free, geometry, 40, plan(39), 256, reserve, 0).is_ok());
        assert!(
            admit_banks(
                reserve + 40 * slot_set,
                geometry,
                40,
                plan(40),
                256,
                reserve,
                0
            )
            .is_ok()
        );
    }
}
