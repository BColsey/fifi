//! Replay captured router inputs through the real CUDA router; no model trajectory is changed.
use super::*;
use crate::{
    family::qwen::flash_next::load::gguf::tests::pinned_router_host_data,
    kernels::cuda::resident::router::{RouterCall, RouterOut, RouterShape, RouterWeights},
};

const LAYER: usize = 3; // The selected pinned fixture boundary, not model geometry.
const TRACE_HASHES: [(&str, &str, &str); 4] = [
    (
        "control",
        "FIFI_FLASH_GGUF_ROUTER_CONTROL",
        "0077ace7b6d8041c1b5d9a9028f3d65d5838927fd9b648b8d7f40d8a7cfb73e6",
    ),
    (
        "first",
        "FIFI_FLASH_GGUF_ROUTER_FIRST",
        "b2a675d3929c52f81fabdfeea8d05c47d4e2692c7bc984de28329974ce913948",
    ),
    (
        "second",
        "FIFI_FLASH_GGUF_ROUTER_SECOND",
        "f39b8b001653211de98bbb4ed10ae9bf146214b3aa990950da7ac06c199930ca",
    ),
    (
        "both",
        "FIFI_FLASH_GGUF_ROUTER_BOTH",
        "18b3aa131e3fbcefd7558f2956c978dcc02ec56603b01d238d6896bd32304434",
    ),
];

struct Input {
    name: &'static str,
    values: Vec<f32>,
    expected_ids: Vec<i32>,
}

#[derive(serde::Serialize)]
struct ResultRow {
    name: &'static str,
    input: Vec<f32>,
    ids: Vec<i32>,
    logits: Vec<f32>,
    weights: Vec<f32>,
    shared_gate: f32,
    cpu_logits: Vec<f32>,
    cpu_ids: Vec<i32>,
    cpu_weights: Vec<f32>,
    cpu_shared_gate: f32,
}

fn capture_host(path: &Path, width: usize, top_k: usize) -> Input {
    let text = std::fs::read_to_string(path).unwrap();
    let rows: Vec<serde_json::Value> = text
        .lines()
        .map(|s| serde_json::from_str(s).unwrap())
        .collect();
    let mut gate = rows
        .iter()
        .filter(|r| r["name"] == "blk.3.ffn_gate_exps.weight");
    let first = gate.next().expect("host route capture");
    let values: Vec<f32> = serde_json::from_value(first["input"].clone()).unwrap();
    let mut ids = vec![first["expert"].as_i64().unwrap() as i32];
    for row in gate {
        let v: Vec<f32> = serde_json::from_value(row["input"].clone()).unwrap();
        assert_eq!(v, values, "host experts require the same router input");
        ids.push(row["expert"].as_i64().unwrap() as i32);
    }
    assert_eq!(ids.len(), top_k);
    let mut unique = ids.clone();
    unique.sort_unstable();
    unique.dedup();
    assert_eq!(unique.len(), top_k);
    assert_eq!(values.len(), width);
    assert!(values.iter().all(|x| x.is_finite()));
    Input {
        name: "host",
        values,
        expected_ids: ids,
    }
}

fn trace_input(name: &'static str, path: &Path, width: usize, top_k: usize) -> Input {
    let rows: Vec<serde_json::Value> =
        serde_json::from_reader(std::fs::File::open(path).unwrap()).unwrap();
    assert_eq!(rows.len(), 15);
    let row = &rows[8];
    assert_eq!(
        (
            row["position"].as_u64(),
            row["token"].as_u64(),
            row["layer"].as_u64()
        ),
        (Some(1), Some(11), Some(LAYER as u64))
    );
    let values: Vec<f32> = serde_json::from_value(row["gpu_router_input"].clone()).unwrap();
    let expected_ids: Vec<i32> = serde_json::from_value(row["gpu_ids"].clone()).unwrap();
    assert_eq!(values.len(), width);
    assert_eq!(expected_ids.len(), top_k);
    assert!(values.iter().all(|x| x.is_finite()));
    Input {
        name,
        values,
        expected_ids,
    }
}

fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn replay(step: &FlashStep<'_>, model: &GgufModel, input: &Input) -> ResultRow {
    let g = &step.geometry;
    let mixed = step.slot(Slot::Mixed(Site::Mlp)).unwrap();
    mixed
        .buffer()
        .write_at(
            step.stream,
            mixed.offset(),
            &input
                .values
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    assert!(same_bits(
        &step.read_f32(mixed, input.values.len()).unwrap(),
        &input.values
    ));
    step.ctx
        .router(
            step.stream,
            RouterCall {
                shape: RouterShape {
                    m: 1,
                    hidden: g.hidden,
                    n_expert: g.experts,
                    top_k: g.top_k,
                },
                x: mixed,
                weights: RouterWeights {
                    w: step.float(&layer_name(LAYER, "mlp.gate.weight")).unwrap(),
                    w_shared: Some(
                        step.float(&layer_name(LAYER, "mlp.shared_expert_gate.weight"))
                            .unwrap(),
                    ),
                },
                logits: step.slot(Slot::RouterLogits).unwrap(),
                out: RouterOut {
                    ids: step.slot(Slot::TopkId).unwrap(),
                    weights: step.slot(Slot::TopkWeight).unwrap(),
                    shared_gate: Some(step.slot(Slot::SharedGate).unwrap()),
                    status: step.status_word().unwrap(),
                },
            },
        )
        .unwrap();
    let logits = step
        .read_f32(step.slot(Slot::RouterLogits).unwrap(), g.experts)
        .unwrap();
    let weights = step
        .read_f32(step.slot(Slot::TopkWeight).unwrap(), g.top_k)
        .unwrap();
    let shared_gate = step
        .read_f32(step.slot(Slot::SharedGate).unwrap(), 1)
        .unwrap()[0];
    let ids_buf = step.slot(Slot::TopkId).unwrap();
    let mut bytes = vec![0; g.top_k * 4];
    ids_buf
        .buffer()
        .read_at(step.stream, ids_buf.offset(), &mut bytes)
        .unwrap();
    let ids: Vec<i32> = bytes
        .chunks_exact(4)
        .map(|b| i32::from_ne_bytes(b.try_into().unwrap()))
        .collect();
    assert_eq!(
        ids, input.expected_ids,
        "replayed live/captured ordered route changed for {}",
        input.name
    );
    assert!(logits.iter().chain(&weights).all(|x| x.is_finite()) && shared_gate.is_finite());
    assert_eq!(step.health().unwrap(), (0, 0));
    let cpu = pinned_router_host_data(model, &input.values, LAYER);
    let cpu_ids: Vec<i32> = cpu.route.iter().map(|(id, _)| *id as i32).collect();
    let cpu_weights: Vec<f32> = cpu.route.iter().map(|(_, weight)| *weight).collect();
    let mut sorted_gpu = ids.clone();
    let mut sorted_cpu = cpu_ids.clone();
    sorted_gpu.sort_unstable();
    sorted_cpu.sort_unstable();
    assert_eq!(
        sorted_gpu, sorted_cpu,
        "same-input CPU/GPU router sets differ for {}",
        input.name
    );
    let max_weight_error = ids
        .iter()
        .zip(&weights)
        .map(|(id, actual)| {
            let index = cpu_ids
                .iter()
                .position(|candidate| candidate == id)
                .unwrap();
            (actual - cpu_weights[index]).abs()
        })
        .fold(0.0_f32, f32::max);
    let gate_error = (shared_gate - cpu.shared_gate).abs();
    assert!(
        max_weight_error < 0.0001 && gate_error < 0.0001,
        "same-input router weight/gate error: weight={max_weight_error}, gate={gate_error}"
    );
    let error = relative_l2(&logits, &cpu.logits);
    assert!(
        error.is_finite() && error < 0.0001,
        "router replay numerical error {error}"
    );
    eprintln!(
        "layer-3 router replay {}: gpu_minus_cpu_logits_relative_l2={error:.9} max_weight_abs={max_weight_error:.9} gate_abs={gate_error:.9} gpu_margin_157_270={:.9} cpu_margin_157_270={:.9} gpu_ids={ids:?} cpu_ids={cpu_ids:?}",
        input.name,
        logits[157] - logits[270],
        cpu.logits[157] - cpu.logits[270]
    );
    ResultRow {
        name: input.name,
        input: input.values.clone(),
        ids,
        logits,
        weights,
        shared_gate,
        cpu_logits: cpu.logits,
        cpu_ids,
        cpu_weights,
        cpu_shared_gate: cpu.shared_gate,
    }
}

#[test]
#[ignore = "requires pinned GGUF, exact intervention traces and an admitted CUDA device"]
fn pinned_layer_three_router_input_replay() {
    let path = std::env::var_os("FIFI_FLASH_GGUF").expect("pinned first shard");
    assert_pinned_file(
        &path,
        "9d5644b371ffac921427c8021cc7242dab8e3ba37d7b48d1e79fe3770fbf9915",
    );
    let capture =
        std::env::var_os("FIFI_FLASH_GGUF_ROUTER_HOST_CAPTURE").expect("pinned host capture");
    assert_pinned_file(
        &capture,
        "896823186ddb50caf9ef6758ca4e8fe85316d541510d40f051fcf5a937e35615",
    );
    let output = std::env::var_os("FIFI_FLASH_GGUF_ROUTER_REPLAY_OUT").expect("new output path");
    assert!(!Path::new(&output).exists());
    let model = GgufModel::open(Path::new(&path)).unwrap().unwrap();
    let shape = model.config().model_shape().unwrap();
    let mut inputs = vec![capture_host(
        Path::new(&capture),
        shape.hyper.hidden(),
        shape.moe.top_k(),
    )];
    for (name, variable, hash) in TRACE_HASHES {
        let path = std::env::var_os(variable).expect(variable);
        assert_pinned_file(&path, hash);
        inputs.push(trace_input(
            name,
            Path::new(&path),
            shape.hyper.hidden(),
            shape.moe.top_k(),
        ));
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    // The native owner requires at least one native matrix; router weights alone are F32.
    let weights =
        CudaWeights::from_native(model.upload_layer_trunk(LAYER, &ctx, &stream).unwrap()).unwrap();
    let step = FlashStep::new_gguf_decode(
        &ctx,
        &stream,
        &weights,
        shape,
        model.config().kinds(),
        3,
        model.map().max_quantized_input_width().unwrap(),
        LAYER..LAYER + 1,
    )
    .unwrap();
    step.begin_token().unwrap();
    let results: Vec<ResultRow> = inputs
        .iter()
        .map(|input| replay(&step, &model, input))
        .collect();
    let host_again = replay(&step, &model, &inputs[0]);
    let control_again = replay(&step, &model, &inputs[1]);
    for (first, again) in [(&results[0], &host_again), (&results[1], &control_again)] {
        assert_eq!(first.ids, again.ids);
        assert!(
            same_bits(&first.logits, &again.logits) && same_bits(&first.weights, &again.weights)
        );
        assert_eq!(first.shared_gate.to_bits(), again.shared_gate.to_bits());
    }
    step.finish_token().unwrap();
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .unwrap();
    serde_json::to_writer(
        file,
        &serde_json::json!({"position":1,"token":11,"layer":LAYER,
            "model_first_shard_sha256":"9d5644b371ffac921427c8021cc7242dab8e3ba37d7b48d1e79fe3770fbf9915",
            "host_capture_sha256":"896823186ddb50caf9ef6758ca4e8fe85316d541510d40f051fcf5a937e35615",
            "trace_sha256":TRACE_HASHES.iter().map(|(name, _, hash)| (*name, *hash)).collect::<std::collections::BTreeMap<_,_>>(),
            "results":results}),
    )
    .unwrap();
    eprintln!(
        "captured host and four live router inputs replayed; host/control repeated bitwise; not propagated model parity"
    );
}
