//! Final mixer and full-vocabulary head on captured host streams, not a CUDA trunk run.
use super::*;
use crate::{
    family::qwen::flash_next::{
        cuda::staged_moe::GgufStep,
        load::gguf::{GgufModel, staged_trunk::GgufTrunk, tests::pinned_final_host_reference},
    },
    kernels::cuda::CudaContext,
};

/// Report payload sizes only. Allocator/context overhead, host allocations, transient staging,
/// and a safety margin must still be admitted; this does not initialize a CUDA device.
#[test]
#[ignore = "requires the verified pinned GGUF checkpoint"]
fn pinned_full_staged_payload_report() {
    use crate::family::qwen::flash_next::{
        cuda::{
            state::{FlashGeometry, StatePlan},
            step_plan::{StepOptions, StepPlan},
        },
        load::gguf::staged_budget::routed_budget,
        moe::context_budget::KvFormat,
    };
    use crate::kernels::cuda::resident::quantize_q8_1::q8_1_bytes;

    let path = std::env::var_os("FIFI_FLASH_GGUF").expect("pinned first shard");
    assert_pinned_file(
        &path,
        "9d5644b371ffac921427c8021cc7242dab8e3ba37d7b48d1e79fe3770fbf9915",
    );
    let model = GgufModel::open(Path::new(&path)).unwrap().unwrap();
    let shape = model.config().model_shape().unwrap();
    let layers = model.config().layers();
    let slots = shape.moe.top_k() + 1;
    let routed = routed_budget(&model, 0..layers, slots).unwrap();
    let geometry = FlashGeometry::new(&shape, model.config().kinds()).unwrap();
    let state = StatePlan::new(&geometry, 5, KvFormat::F16).unwrap();
    let sm_count: u32 = std::env::var("FIFI_FLASH_GGUF_REPORT_SM_COUNT")
        .expect("selected device SM count")
        .parse()
        .unwrap();
    assert!(sm_count > 0);
    let step = StepPlan::gguf_decode(
        &geometry,
        5,
        &StepOptions::for_sm_count(sm_count),
        model.map().max_quantized_input_width().unwrap(),
    )
    .unwrap();
    let ids = shape.moe.top_k() * size_of::<i32>();
    let qx = q8_1_bytes(shape.hyper.hidden(), 1).unwrap();
    let qa = q8_1_bytes(shape.moe.inter(), shape.moe.top_k()).unwrap();
    let zeros = shape.hyper.hidden() * size_of::<f32>();
    let pack = super::super::step::activation_pack_bytes(&geometry).unwrap();
    let trunk = model.staged_sizes().unwrap().trunk_bytes;
    eprintln!(
        "pinned full staged payload: layers={layers} slots={slots} positions=5 sm_count={sm_count} mapped_device_trunk_bytes={trunk} bank_stack_bytes={} pinned_record_bytes={} state_arena_bytes={} step_arena_bytes={} routed_scratch_bytes={} zero_bytes={zeros} activation_pack_bytes={pack}; excludes allocation overhead, host feed, transient staging and safety margin; NOT admission",
        routed.stacks,
        routed.pinned_records,
        state.arena_bytes(),
        step.arena_bytes(),
        ids + qx + qa,
    );
}

fn greedy(row: &[f32]) -> usize {
    assert!(!row.is_empty() && row.iter().all(|v| v.is_finite()));
    let mut best = 0;
    for i in 1..row.len() {
        if row[i] > row[best] {
            best = i;
        }
    }
    best
}

/// Repeated warm-buffer timing. Full-tail samples include a device-to-device stream restore.
fn benchmark_tail(step: &FlashStep<'_>, input: &[f32], expected: &[f32]) {
    let stream_slot = step.slot(Slot::Stream).unwrap();
    let mixed_slot = step.slot(Slot::Mixed(Site::Final)).unwrap();
    let logits_slot = step.slot(Slot::Logits).unwrap();
    let live_stream = step.read_f32(stream_slot, input.len()).unwrap();
    let overlaps = std::ptr::eq(stream_slot.buffer(), logits_slot.buffer())
        && stream_slot.offset() < logits_slot.offset() + logits_slot.bytes()
        && logits_slot.offset() < stream_slot.offset() + stream_slot.bytes();
    eprintln!(
        "final-head arena diagnostic: stream_logits_overlap={overlaps} stream_preserved_after_head={} stream_offset={} logits_offset={}",
        live_stream == input,
        stream_slot.offset(),
        logits_slot.offset()
    );
    let bytes: Vec<u8> = input.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let saved = step.ctx.upload(&bytes).unwrap();
    let source = saved.range(0, bytes.len()).unwrap();
    let destination = stream_slot
        .buffer()
        .range(stream_slot.offset(), bytes.len())
        .unwrap();
    let mixed = step.read_f32(mixed_slot, step.geometry.hidden).unwrap();
    let run = |tail| {
        if tail {
            step.ctx.copy_device(&step.stream, source, destination)?;
            step.head(1)
        } else {
            step.project("lm_head", mixed_slot, logits_slot, 1)
        }
    };
    const REPETITIONS: usize = 25;
    for tail in [false, true, true, false] {
        for _ in 0..5 {
            run(tail).unwrap();
        }
        let start = step.ctx.event_timing().unwrap();
        let end = step.ctx.event_timing().unwrap();
        step.stream.record_timing(&start).unwrap();
        for _ in 0..REPETITIONS {
            run(tail).unwrap();
        }
        step.stream.record_timing(&end).unwrap();
        step.stream.synchronize().unwrap();
        let ms = start.elapsed_ms_since(&end).unwrap() / REPETITIONS as f32;
        assert!(ms.is_finite() && ms > 0.0);
        let got = step
            .read_f32(step.slot(Slot::Logits).unwrap(), expected.len())
            .unwrap();
        let same_logits = got == expected;
        let same_mixed = step.read_f32(mixed_slot, mixed.len()).unwrap() == mixed;
        assert!(
            same_logits && same_mixed,
            "timing changed logits or mixed input; logit relative L2={}",
            relative_l2(&got, expected)
        );
        eprintln!(
            "pinned GGUF warm tail benchmark: mode={} repeats={REPETITIONS} ms_per_call={ms:.6} (device events; excludes trunk, host transfers and sampling)",
            if tail {
                "restore+mixer+head"
            } else {
                "head+quantize"
            }
        );
    }
    step.finish_token().unwrap();
}

#[test]
#[ignore = "requires pinned GGUF, pinned prompt-final streams and an admitted CUDA device"]
fn pinned_prompt_final_mixer_and_head_match_host() {
    let path = std::env::var_os("FIFI_FLASH_GGUF").expect("pinned first shard");
    let streams_path = std::env::var_os("FIFI_FLASH_GGUF_FINAL_STREAMS").expect("host streams");
    let streams_sha =
        std::env::var("FIFI_FLASH_GGUF_FINAL_STREAMS_SHA256").expect("streams sha256");
    let oracle_path = std::env::var_os("FIFI_FLASH_GGUF_HOST_ORACLE").expect("host oracle");
    // The three remaining shards are checked against the publisher manifest by the runner.
    assert_pinned_file(
        &path,
        "9d5644b371ffac921427c8021cc7242dab8e3ba37d7b48d1e79fe3770fbf9915",
    );
    assert_pinned_file(
        &oracle_path,
        "2c2133dccb289ad6f3e244d304f655189c04497bf19cd3b609779eb5af235ffa",
    );
    assert_pinned_file(&streams_path, &streams_sha);
    let streams: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(streams_path).unwrap()).unwrap();
    let oracle: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(oracle_path).unwrap()).unwrap();
    // The prompt is defined by the fixtures, not here; the two must agree.
    let prompt: Vec<u32> = serde_json::from_value(oracle["prompt"].clone()).unwrap();
    assert!(!prompt.is_empty(), "oracle prompt is empty");
    assert_eq!(streams["prompt"], serde_json::json!(prompt));
    let inputs: Vec<Vec<f32>> = serde_json::from_value(streams["streams"].clone()).unwrap();
    let oracle_logits: Vec<f32> = serde_json::from_value(oracle["logits"][0].clone()).unwrap();
    assert_eq!(inputs.len(), prompt.len());
    let model = GgufModel::open(Path::new(&path)).unwrap().unwrap();
    let shape = model.config().model_shape().unwrap();
    assert_eq!(
        streams["layers"].as_u64().unwrap() as usize,
        model.config().layers()
    );
    assert!(inputs.iter().all(|row| row.len() == shape.hyper.width()));
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let other = CudaContext::new().unwrap();
    let foreign = other.stream().unwrap();
    let before = ctx.live_bytes();
    assert!(model.upload_final_trunk(&ctx, &foreign).is_err());
    assert_eq!(ctx.live_bytes(), before);
    drop(foreign);
    drop(other);
    let trunk = GgufTrunk::upload_with_final(&model, &ctx, &stream, 0..1).unwrap();
    let bound_step = GgufStep::new(&ctx, &stream, &trunk, 1).unwrap();
    let step = bound_step.step();
    eprintln!(
        "staged model-bound layer/final mixer/head resident device bytes={}",
        ctx.live_bytes()
    );
    // Exercise the oracle-bound prompt-final position first so an earlier-token diagnostic
    // cannot prevent that comparison from running.
    for position in (0..inputs.len()).rev() {
        let input = &inputs[position];
        let (host_mixed, host_logits) = pinned_final_host_reference(&model, input);
        assert_eq!(host_logits.len(), shape.vocab);
        if position == prompt.len() - 1 {
            let error = relative_l2(&host_logits, &oracle_logits);
            assert!(
                error < 1e-6,
                "reconstructed host prompt logits differ from oracle: {error}"
            );
            assert_eq!(
                greedy(&host_logits) as u64,
                oracle["greedy"][0].as_u64().unwrap()
            );
        }
        step.begin_token().unwrap();
        let state = step.slot(Slot::Stream).unwrap();
        let bytes: Vec<u8> = input.iter().flat_map(|v| v.to_ne_bytes()).collect();
        state
            .buffer()
            .write_at(&stream, state.offset(), &bytes)
            .unwrap();
        step.head(1).unwrap();
        step.finish_token().unwrap();
        let mixed = step
            .read_f32(
                step.slot(Slot::Mixed(Site::Final)).unwrap(),
                host_mixed.len(),
            )
            .unwrap();
        let logits = step
            .read_f32(step.slot(Slot::Logits).unwrap(), shape.vocab)
            .unwrap();
        assert!(mixed.iter().chain(&logits).all(|v| v.is_finite()));
        let mixed_error = relative_l2(&mixed, &host_mixed);
        let logit_error = relative_l2(&logits, &host_logits);
        let (gpu_top, host_top) = (greedy(&logits), greedy(&host_logits));
        eprintln!(
            "captured-host-stream head position={position} mixed_rel_l2={mixed_error:.9} logits_rel_l2={logit_error:.9} gpu_top={gpu_top} host_top={host_top}"
        );
        assert!(mixed_error < 0.01, "final mixer error {mixed_error}");
        assert!(
            logit_error < 0.05,
            "full-vocabulary logit error {logit_error}"
        );
        assert_eq!(
            gpu_top, host_top,
            "greedy differs at prompt position {position}"
        );
        if position == prompt.len() - 1
            && std::env::var_os("FIFI_FLASH_GGUF_TEST_HEAD_BENCH").is_some()
        {
            benchmark_tail(step, input, &logits);
        }
    }
}
