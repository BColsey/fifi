//! Checkpoint geometry and text memory plan for the fixed-loop Inkling 2 Mini release.

use std::{collections::BTreeMap, fs::File, ops::Range, path::Path};

use safetensors::{Dtype, tensor::TensorInfo};
use serde::Deserialize;

use super::{
    kv_format::{KvFormat, KvRowShape},
    plan_decision::{PlanDecision, PlanFixity},
};
use crate::{EngError, checkpoint_json::UniqueValue, weights::read_stored_header};

pub(crate) const FAMILY: &str = "inkling2_mini";
pub(crate) const MODEL_TYPE: &str = "inkling2_loop";
const SHARD: &str = "model-00001-of-00001.safetensors";
const LAYERS: usize = 24;
const VISITS: usize = 48;
const CORE_START: usize = 6;
const CORE_END: usize = 18;
const PASSES: usize = 3;
const EXPERTS: usize = 128;
const TOP_K: usize = 8;
const HIDDEN: usize = 2304;
const INTERMEDIATE: usize = 1152;
const QUERY_HEADS: usize = 24;
const KV_HEADS: u64 = 8;
const HEAD_DIM: u64 = 128;
const WINDOW: u32 = 1024;
const MAX_CONTEXT: u32 = 1_048_576;
const CONV_TAPS: u64 = 4;
const HEADER_LIMIT: u64 = 64 << 20;

#[derive(Deserialize)]
struct ModelConfig {
    model_type: String,
    architectures: Vec<String>,
    text_config: TextConfig,
    quantization_config: QuantizationConfig,
}

#[derive(Deserialize)]
struct TextConfig {
    model_type: String,
    num_hidden_layers: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_attention_heads: usize,
    num_key_value_heads: u64,
    head_dim: u64,
    swa_num_attention_heads: usize,
    swa_num_key_value_heads: u64,
    swa_head_dim: u64,
    vocab_size: usize,
    unpadded_vocab_size: usize,
    model_max_length: u32,
    local_layer_ids: Vec<usize>,
    sliding_window_size: u32,
    sconv_kernel_size: u64,
    n_layers_per_attn_res_block: usize,
    n_routed_experts: usize,
    num_experts_per_tok: usize,
    n_shared_experts: usize,
    loop_config: LoopConfig,
}

#[derive(Deserialize)]
struct LoopConfig {
    core_start: usize,
    core_end: usize,
    num_passes: usize,
    gate_rank: usize,
    is_varloop: bool,
}

#[derive(Deserialize)]
struct QuantizationConfig {
    quant_method: String,
    quant_algo: String,
}

fn checkpoint_error(detail: impl std::fmt::Display) -> EngError {
    EngError::Checkpoint(format!("Inkling 2 Mini: {detail}"))
}

fn parse_json<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Result<T, EngError> {
    let value = serde_json::from_slice::<UniqueValue>(bytes)
        .map_err(checkpoint_error)?
        .0;
    serde_json::from_value(value).map_err(checkpoint_error)
}

fn validate_config(bytes: &[u8]) -> Result<(), EngError> {
    let config: ModelConfig = parse_json(bytes)?;
    let text = config.text_config;
    let loop_config = text.loop_config;
    if config.model_type != MODEL_TYPE
        || config.architectures != ["Inkling2LoopForConditionalGeneration"]
        || text.model_type != "inkling2_loop_text"
        || (
            loop_config.core_start,
            loop_config.core_end,
            loop_config.num_passes,
        ) != (CORE_START, CORE_END, PASSES)
        || loop_config.gate_rank != 64
        || loop_config.is_varloop
    {
        return Err(checkpoint_error("expected the fixed three-pass Mini loop"));
    }
    let local = (0..LAYERS).filter(|layer| is_local(*layer));
    if text.num_hidden_layers != LAYERS
        || text.hidden_size != HIDDEN
        || text.intermediate_size != INTERMEDIATE
        || text.num_attention_heads != QUERY_HEADS
        || text.num_key_value_heads != KV_HEADS
        || text.head_dim != HEAD_DIM
        || text.swa_num_attention_heads != QUERY_HEADS
        || text.swa_num_key_value_heads != KV_HEADS
        || text.swa_head_dim != HEAD_DIM
        || text.vocab_size != 201_024
        || text.unpadded_vocab_size != 200_058
        || text.model_max_length != MAX_CONTEXT
        || !text.local_layer_ids.into_iter().eq(local)
        || text.sliding_window_size != WINDOW
        || text.sconv_kernel_size != CONV_TAPS
        || text.n_layers_per_attn_res_block != 6
        || text.n_routed_experts != EXPERTS
        || text.num_experts_per_tok != TOP_K
        || text.n_shared_experts != 0
        || config.quantization_config.quant_method != "modelopt"
        || config.quantization_config.quant_algo != "NVFP4"
    {
        return Err(checkpoint_error(
            "checkpoint geometry differs from the released Mini",
        ));
    }
    Ok(())
}

fn is_local(layer: usize) -> bool {
    layer % 6 != 5
}

/// The weight index can repeat. KV and short-convolution buffers use the unique visit index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LayerVisit {
    weight_layer: usize,
    state_slot: usize,
    pass: Option<usize>,
}

fn visits() -> [LayerVisit; VISITS] {
    let width = CORE_END - CORE_START;
    std::array::from_fn(|state_slot| {
        let (weight_layer, pass) = if state_slot < CORE_START {
            (state_slot, None)
        } else if state_slot < CORE_START + width * PASSES {
            let core_visit = state_slot - CORE_START;
            (CORE_START + core_visit % width, Some(core_visit / width))
        } else {
            (state_slot - width * (PASSES - 1), None)
        };
        LayerVisit {
            weight_layer,
            state_slot,
            pass,
        }
    })
}

#[derive(Deserialize)]
struct WeightIndex {
    metadata: IndexMetadata,
    weight_map: BTreeMap<String, String>,
}

#[derive(Deserialize)]
struct IndexMetadata {
    total_size: u64,
}

#[derive(Default)]
struct WeightPlan {
    text_resident: u64,
    vision: u64,
    expert_storage: [u64; LAYERS],
    routed_per_layer: [u64; LAYERS],
    coalesced_reads: [usize; LAYERS],
    scattered_reads: [usize; LAYERS],
}

fn tensor_bytes(tensor: &TensorInfo) -> u64 {
    (tensor.data_offsets.1 - tensor.data_offsets.0) as u64
}

enum ExpertRegion {
    Slices { start: u64, stride: u64 },
    Shared(Range<u64>),
}

/// Sort by the physical expert slot and merge only adjacent selected slots.
fn expert_read_runs(
    regions: &[ExpertRegion],
    experts: &[usize],
) -> Result<Vec<Range<u64>>, EngError> {
    if experts.is_empty() || experts.iter().any(|&expert| expert >= EXPERTS) {
        return Err(EngError::BadRequest(
            "Inkling expert selection is empty or out of range".into(),
        ));
    }
    let mut sorted = experts.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    let mut ranges = Vec::new();
    for region in regions {
        match region {
            ExpertRegion::Slices { start, stride } => {
                for run in sorted.chunk_by(|&left, &right| left + 1 == right) {
                    let first = run[0] as u64;
                    let end = first + run.len() as u64;
                    ranges.push((start + first * stride)..(start + end * stride));
                }
            }
            ExpertRegion::Shared(range) => ranges.push(range.clone()),
        }
    }
    Ok(ranges)
}

fn expert_regions(
    entries: &BTreeMap<String, TensorInfo>,
    data_start: u64,
    layer: usize,
) -> Result<Vec<ExpertRegion>, EngError> {
    let mut regions = Vec::new();
    let nvfp4 = layer < 20;
    let group = if nvfp4 { 16 } else { 32 };
    // w13 packs gate and up rows together. w2 projects to the hidden dimension.
    for (projection, rows, cols) in [
        ("w13", INTERMEDIATE * 2, HIDDEN),
        ("w2", HIDDEN, INTERMEDIATE),
    ] {
        let stem = format!("model.llm.layers.{layer}.mlp.experts.{projection}_weight");
        let mut get = |suffix: &str, dtype: Dtype, shape: &[usize], per_expert: bool| {
            let key = format!("{stem}{suffix}");
            let info = entries
                .get(&key)
                .ok_or_else(|| checkpoint_error(format!("missing {key}")))?;
            if info.dtype != dtype || info.shape != shape {
                return Err(checkpoint_error(format!(
                    "unexpected storage layout for {key}"
                )));
            }
            let start = data_start + info.data_offsets.0 as u64;
            let stored = tensor_bytes(info);
            let region = if per_expert {
                ExpertRegion::Slices {
                    start,
                    stride: stored / EXPERTS as u64,
                }
            } else {
                ExpertRegion::Shared(start..start + stored)
            };
            regions.push(region);
            Ok(())
        };
        if nvfp4 {
            get("", Dtype::U8, &[EXPERTS, rows, cols / 2], true)?;
            get(
                "_scale",
                Dtype::F8_E4M3,
                &[EXPERTS, rows, cols / group],
                true,
            )?;
            get("_amax", Dtype::F32, &[EXPERTS], true)?;
            get("_outer_scale_denominator", Dtype::F32, &[1], false)?;
        } else {
            get("", Dtype::F8_E4M3, &[EXPERTS, rows, cols], true)?;
            get("_scale", Dtype::U8, &[EXPERTS, rows, cols / group], true)?;
        }
    }
    Ok(regions)
}

fn weight_plan(
    index: &WeightIndex,
    data_start: u64,
    entries: BTreeMap<String, TensorInfo>,
) -> Result<WeightPlan, EngError> {
    if index.weight_map.len() != entries.len()
        || index
            .weight_map
            .iter()
            .any(|(key, file)| file != SHARD || !entries.contains_key(key))
        || index.metadata.total_size != entries.values().map(tensor_bytes).sum::<u64>()
    {
        return Err(checkpoint_error("index and shard inventory disagree"));
    }
    let mut plan = WeightPlan::default();
    let mut expert_tensors = 0;
    for (key, tensor) in &entries {
        let bytes = tensor_bytes(tensor);
        if key.starts_with("model.visual.") {
            plan.vision += bytes;
        } else if key.starts_with("model.llm.") {
            if key.contains(".mlp.experts.") {
                let layer = key
                    .strip_prefix("model.llm.layers.")
                    .and_then(|name| name.split_once(".mlp.experts."))
                    .and_then(|(layer, _)| layer.parse::<usize>().ok())
                    .filter(|&layer| layer < LAYERS)
                    .ok_or_else(|| checkpoint_error(format!("unknown expert layer in {key}")))?;
                plan.expert_storage[layer] += bytes;
                expert_tensors += 1;
            } else {
                plan.text_resident += bytes;
            }
        } else {
            return Err(checkpoint_error(format!(
                "unknown tensor namespace in {key}"
            )));
        }
    }
    if expert_tensors != 20 * 8 + 4 * 4 {
        return Err(checkpoint_error("unexpected number of expert tensors"));
    }
    let adjacent: [usize; TOP_K] = std::array::from_fn(|i| i);
    let spread: [usize; TOP_K] = std::array::from_fn(|i| 2 * i);
    for layer in 0..LAYERS {
        let regions = expert_regions(&entries, data_start, layer)?;
        let grouped = expert_read_runs(&regions, &adjacent)?;
        let scattered = expert_read_runs(&regions, &spread)?;
        plan.routed_per_layer[layer] = grouped.iter().map(|range| range.end - range.start).sum();
        plan.coalesced_reads[layer] = grouped.len();
        plan.scattered_reads[layer] = scattered.len();
    }
    Ok(plan)
}

#[derive(Debug, PartialEq, Eq)]
struct StatePlan {
    kv: u64,
    convolution: u64,
    local: usize,
    full: usize,
}

fn state_plan(context: u32) -> Result<StatePlan, EngError> {
    if !(2..=MAX_CONTEXT).contains(&context) {
        return Err(EngError::BadRequest(format!(
            "Inkling 2 Mini context must be between 2 and {MAX_CONTEXT} tokens"
        )));
    }
    let local = visits()
        .iter()
        .filter(|visit| is_local(visit.weight_layer))
        .count();
    let full = VISITS - local;
    let row = KvRowShape {
        heads_kv: KV_HEADS,
        head_dim: HEAD_DIM,
    };
    let layer_bytes = KvFormat::Bf16.layer_position_bytes(row)?;
    let kv = layer_bytes
        * (local as u64 * u64::from(context.min(WINDOW)) + full as u64 * u64::from(context));
    // The K/V convolutions each keep a KV plane's channels. The attention and MLP
    // output convolutions each keep hidden channels. All four retain taps - 1 rows.
    let channels = 2 * (KV_HEADS * HEAD_DIM + HIDDEN as u64);
    let convolution = VISITS as u64 * channels * (CONV_TAPS - 1) * 2;
    Ok(StatePlan {
        kv,
        convolution,
        local,
        full,
    })
}

pub(crate) fn inspect(dir: &Path, context: Option<u32>) -> Result<Vec<PlanDecision>, EngError> {
    let guard = crate::storage_guard::LocalGuard::new()?;
    let config_path = dir.join("config.json");
    let index_path = dir.join("model.safetensors.index.json");
    let shard_path = dir.join(SHARD);
    for path in [&config_path, &index_path, &shard_path] {
        guard.check(path, "model weights")?;
    }
    validate_config(&std::fs::read(config_path).map_err(checkpoint_error)?)?;
    let index = parse_json(&std::fs::read(index_path).map_err(checkpoint_error)?)?;
    let mut shard = File::open(shard_path).map_err(checkpoint_error)?;
    let len = shard.metadata().map_err(checkpoint_error)?.len();
    let (data_start, entries) = read_stored_header(&mut shard, len, HEADER_LIMIT)?;
    let weights = weight_plan(&index, data_start, entries.into_iter().collect())?;
    let context = context.unwrap_or(MAX_CONTEXT);
    let state = state_plan(context)?;
    let routed_peak = weights
        .routed_per_layer
        .into_iter()
        .max()
        .expect("24 layers");
    let schedule = visits();
    let state_slots = schedule[VISITS - 1].state_slot + 1;
    let core_visits = schedule.iter().filter(|visit| visit.pass.is_some()).count();
    let routed_uncached: u64 = schedule
        .iter()
        .map(|visit| weights.routed_per_layer[visit.weight_layer])
        .sum();
    let core_resident: u64 = weights.expert_storage[CORE_START..CORE_END].iter().sum();
    let outside_core: u64 = schedule
        .iter()
        .filter(|visit| visit.pass.is_none())
        .map(|visit| weights.routed_per_layer[visit.weight_layer])
        .sum();
    let expert_storage: u64 = weights.expert_storage.iter().sum();
    let row = |name: &str, chosen: String, fixed_at| PlanDecision {
        name: name.into(),
        chosen,
        alternatives: Vec::new(),
        fixed_at,
    };
    Ok(vec![
        row(
            "decoder loop",
            format!(
                "{LAYERS} stored layers, {VISITS} visits ({core_visits} in the core), {state_slots} state slots"
            ),
            PlanFixity::Load,
        ),
        row(
            "text weight inventory",
            format!(
                "{} B non-expert text, {} B routed experts (20 NVFP4 layers, 4 MXFP8)",
                weights.text_resident, expert_storage
            ),
            PlanFixity::Load,
        ),
        row(
            "vision weights",
            format!("{} B outside the text memory plan", weights.vision),
            PlanFixity::Load,
        ),
        row(
            "text state",
            format!(
                "{} B BF16 KV at {context} tokens ({} full, {} local visits), {} B BF16 convolution history",
                state.kv, state.full, state.local, state.convolution
            ),
            PlanFixity::Start,
        ),
        row(
            "exact expert read ranges",
            format!(
                "adjacent versus scattered top-8 IDs need {} vs {} ranges for NVFP4 and {} vs {} for MXFP8",
                weights.coalesced_reads[0],
                weights.scattered_reads[0],
                weights.coalesced_reads[20],
                weights.scattered_reads[20],
            ),
            PlanFixity::Load,
        ),
        row(
            "loop-core residency",
            format!(
                "{core_resident} B of stored core experts once, {outside_core} B raw expert demand per token at uncached outer visits"
            ),
            PlanFixity::Load,
        ),
        row(
            "routed expert reads",
            format!(
                "{routed_peak} B peak raw top-8 layer stage, {routed_uncached} B per token if every visit misses"
            ),
            PlanFixity::Load,
        ),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_core_pass_has_private_state_and_shared_weight_indices() {
        let schedule = visits();
        let expected = (0..6).chain((6..18).cycle().take(36)).chain(18..24);
        assert!(schedule.iter().map(|visit| visit.weight_layer).eq(expected));
        for (i, visit) in schedule.iter().enumerate() {
            assert_eq!(visit.state_slot, i);
            assert_eq!(visit.pass, (6..42).contains(&i).then(|| (i - 6) / 12));
        }
    }

    #[test]
    fn cache_charges_repeat_state_but_bound_local_context() {
        let short = state_plan(512).unwrap();
        assert_eq!(short.kv, 48 * 512 * 4096);
        let long = state_plan(4096).unwrap();
        assert_eq!(long.kv, 40 * 1024 * 4096 + 8 * 4096 * 4096);
        assert_eq!(long.convolution, 1_916_928);
        assert_eq!((long.local, long.full), (40, 8));
        assert!(state_plan(MAX_CONTEXT + 1).is_err());
    }

    #[test]
    fn expert_stage_prices_nvfp4_and_mxfp8_without_repeating_weight_storage() {
        let mut nv = BTreeMap::new();
        let mut mx = BTreeMap::new();
        for (projection, cols) in [("w13", HIDDEN), ("w2", INTERMEDIATE)] {
            let rows = HIDDEN;
            let mut add = |layer, suffix, dtype, shape: Vec<usize>, bytes_per_element| {
                let bytes = shape.iter().product::<usize>() * bytes_per_element;
                let table = if layer == 0 { &mut nv } else { &mut mx };
                table.insert(
                    format!("model.llm.layers.{layer}.mlp.experts.{projection}_weight{suffix}"),
                    TensorInfo {
                        dtype,
                        shape,
                        data_offsets: (0, bytes),
                    },
                );
            };
            add(0, "", Dtype::U8, vec![EXPERTS, rows, cols / 2], 1);
            add(
                0,
                "_scale",
                Dtype::F8_E4M3,
                vec![EXPERTS, rows, cols / 16],
                1,
            );
            add(0, "_amax", Dtype::F32, vec![EXPERTS], 4);
            add(0, "_outer_scale_denominator", Dtype::F32, vec![1], 4);
            add(20, "", Dtype::F8_E4M3, vec![EXPERTS, rows, cols], 1);
            add(20, "_scale", Dtype::U8, vec![EXPERTS, rows, cols / 32], 1);
        }
        let adjacent = [7, 6, 5, 4, 3, 2, 1, 0, 0];
        let scattered = [0, 2, 4, 6, 8, 10, 12, 127];
        for (table, layer, bytes, grouped, individual) in
            [(&nv, 0, 35_831_880, 8, 50), (&mx, 20, 65_691_648, 4, 32)]
        {
            let regions = expert_regions(table, 1024, layer).unwrap();
            for (ids, count) in [
                (adjacent.as_slice(), grouped),
                (scattered.as_slice(), individual),
            ] {
                let runs = expert_read_runs(&regions, ids).unwrap();
                assert_eq!(runs.len(), count);
                assert_eq!(
                    runs.iter().map(|run| run.end - run.start).sum::<u64>(),
                    bytes
                );
                assert!(runs.iter().all(|run| run.start >= 1024));
            }
            assert!(expert_read_runs(&regions, &[128]).is_err());
        }
        assert!(expert_regions(&mx, 0, 0).is_err());
        nv.remove("model.llm.layers.0.mlp.experts.w2_weight_amax");
        assert!(expert_regions(&nv, 0, 0).is_err());
    }
}
