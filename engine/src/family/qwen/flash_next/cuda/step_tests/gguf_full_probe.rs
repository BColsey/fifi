//! Exploratory full-model staged GGUF probe; not production admission or startup.
#[path = "gguf_full_probe/admission.rs"]
mod admission;
#[path = "gguf_full_probe/router_replay.rs"]
mod router_replay;
#[path = "gguf_full_probe/trace.rs"]
mod trace;
#[path = "gguf_full_probe/upstream_intervention.rs"]
mod upstream_intervention;

use super::*;
use crate::family::qwen::flash_next::{
    cuda::{
        driver::TokenFeeds,
        gguf_moe::{GgufMoeOwner, GgufStep},
    },
    load::gguf::{GgufModel, token_feed::GgufTokenFeeds, trunk::GgufTrunk},
};
use crate::kernels::cuda::{CudaContext, resident::step_params::StepParams};
use admission::ProbeBudget;

fn greedy(values: &[f32]) -> usize {
    assert!(!values.is_empty() && values.iter().all(|x| x.is_finite()));
    let mut best = 0;
    for i in 1..values.len() {
        if values[i] > values[best] {
            best = i;
        }
    }
    best
}

#[test]
#[ignore = "requires verified pinned GGUF and an admitted CUDA device"]
fn pinned_guarded_one_layer_and_head() {
    let path = std::env::var_os("FIFI_FLASH_GGUF").expect("pinned first shard");
    assert_pinned_file(
        &path,
        "9d5644b371ffac921427c8021cc7242dab8e3ba37d7b48d1e79fe3770fbf9915",
    );
    let model = GgufModel::open(Path::new(&path)).unwrap().unwrap();
    ProbeBudget::check_host(&model).unwrap();
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let budget = ProbeBudget::new(&model, &ctx, 3).unwrap();
    let trunk =
        GgufTrunk::upload_with_final_guarded(&model, &ctx, &stream, 0..1, |layer, weights| {
            budget.group_guard(&ctx, layer, weights)
        })
        .unwrap();
    budget.before_step(&ctx).unwrap();
    let prior = ctx.live_bytes();
    let _step = GgufStep::new(&ctx, &stream, &trunk, 3).unwrap();
    assert_eq!(ctx.live_bytes() - prior, budget.step_payload());
    budget.before_execution(&ctx).unwrap();
    eprintln!("guarded one-layer plus final staged payload verified");
}

#[test]
#[ignore = "requires pinned GGUF, full host oracle and guarded resource admission"]
fn pinned_full_model_first_logits() {
    run_full_model(None);
}

#[test]
#[ignore = "requires pinned GGUF, host first-token trace and guarded resource admission"]
fn pinned_first_token_full_layer_trace() {
    run_full_model(Some(TraceWindow::All));
}

#[test]
#[ignore = "requires pinned GGUF, host first-token trace and guarded resource admission"]
fn pinned_first_four_layer_trace() {
    run_full_model(Some(TraceWindow::Four));
}

#[test]
#[ignore = "requires pinned GGUF, host first-token trace and guarded resource admission"]
fn pinned_first_six_layer_router_trace() {
    run_full_model(Some(TraceWindow::Six));
}

#[test]
#[ignore = "requires pinned GGUF and guarded resource admission"]
fn pinned_prompt_stateful_four_layer_trace() {
    run_full_model(Some(TraceWindow::StatefulFour));
}

#[test]
#[ignore = "requires pinned GGUF and guarded resource admission"]
fn pinned_prompt_stateful_full_layer_trace() {
    run_full_model(Some(TraceWindow::StatefulAll));
}

#[derive(Clone, Copy)]
enum TraceWindow {
    All,
    Four,
    Six,
    StatefulFour,
    StatefulAll,
}

fn run_full_model(trace_window: Option<TraceWindow>) {
    let path = std::env::var_os("FIFI_FLASH_GGUF").expect("pinned first shard");
    let oracle_path = std::env::var_os("FIFI_FLASH_GGUF_HOST_ORACLE").expect("full host oracle");
    assert_pinned_file(
        &path,
        "9d5644b371ffac921427c8021cc7242dab8e3ba37d7b48d1e79fe3770fbf9915",
    );
    assert_pinned_file(
        &oracle_path,
        "2c2133dccb289ad6f3e244d304f655189c04497bf19cd3b609779eb5af235ffa",
    );
    let oracle: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(oracle_path).unwrap()).unwrap();
    let prompt = [9707u32, 11, 1879];
    assert_eq!(oracle["prompt"], serde_json::json!(prompt));
    let expected: Vec<f32> = serde_json::from_value(oracle["logits"][0].clone()).unwrap();
    let model = GgufModel::open(Path::new(&path)).unwrap().unwrap();
    let shape = model.config().model_shape().unwrap();
    let layers = model.config().layers();
    let end = match trace_window {
        Some(TraceWindow::Four | TraceWindow::StatefulFour) => 4.min(layers),
        Some(TraceWindow::Six) => 6.min(layers),
        _ => layers,
    };
    let trace_stateful = matches!(
        trace_window,
        Some(TraceWindow::StatefulFour | TraceWindow::StatefulAll)
    );
    if trace_stateful && end == layers {
        assert!(
            std::env::var_os("FIFI_FLASH_GGUF_FULL_PROBE_LOGITS_OUT").is_some(),
            "full stateful trace requires a separate logits receipt"
        );
    }
    let trace_first_token = trace_window.is_some() && !trace_stateful;
    let mut stateful = trace_stateful.then(|| {
        trace::StatefulTrace::new(
            end,
            shape.hyper.hidden() * shape.hyper.branches(),
            shape.moe.top_k(),
            shape.moe.experts(),
        )
    });
    let mut trace = trace_first_token.then(|| {
        trace::Trace::open(
            layers,
            shape.hyper.hidden() * shape.hyper.branches(),
            shape.moe.top_k(),
            shape.moe.experts(),
        )
    });
    ProbeBudget::check_host(&model).expect("probe host admission");
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let budget = ProbeBudget::new(&model, &ctx, 3).expect("probe preflight");
    let trunk =
        GgufTrunk::upload_with_final_guarded(&model, &ctx, &stream, 0..end, |layer, weights| {
            budget.group_guard(&ctx, layer, weights)
        })
        .expect("guarded trunk upload");
    let ngram = trunk.model().ngram().unwrap();
    let mut feed = GgufTokenFeeds::new(trunk.model(), &ngram, 3).unwrap();
    budget.before_step(&ctx).unwrap();
    let before_step = ctx.live_bytes();
    let bound = GgufStep::new(&ctx, &stream, &trunk, 3).unwrap();
    assert_eq!(
        ctx.live_bytes() - before_step,
        budget.step_payload(),
        "step allocation disagrees with planned payload"
    );
    let step = bound.step();
    budget.before_bank(&model, &ctx).unwrap();
    let before_bank = (ctx.live_bytes(), ctx.live_owners().pinned);
    let mut owner = GgufMoeOwner::new(&bound, shape.moe.top_k() + 1).unwrap();
    let (device_bank, pinned_bank) = (
        ctx.live_bytes() - before_bank.0,
        ctx.live_owners().pinned - before_bank.1,
    );
    if end == layers {
        assert_eq!(
            device_bank,
            budget.bank_payload().unwrap(),
            "bank device payload differs from plan"
        );
        assert_eq!(
            pinned_bank,
            budget.pinned_bytes(),
            "bank pinned payload differs from plan"
        );
    } else {
        assert!(
            device_bank < budget.bank_payload().unwrap() && pinned_bank < budget.pinned_bytes()
        );
    }
    budget.before_execution(&ctx).unwrap();
    for (position, token) in prompt.into_iter().enumerate() {
        let start = feed.begin(token).unwrap();
        step.begin_token().unwrap();
        let (words, at) = step.words_buffer().unwrap();
        words
            .write_at(
                &stream,
                at,
                &StepParams::new(token as i32, position as i32)
                    .unwrap()
                    .to_bytes(),
            )
            .unwrap();
        let state = step.slot(Slot::Stream).unwrap();
        let write = |values: &[f32]| {
            state
                .buffer()
                .write_at(
                    &stream,
                    state.offset(),
                    &values
                        .iter()
                        .flat_map(|v| v.to_ne_bytes())
                        .collect::<Vec<_>>(),
                )
                .unwrap()
        };
        write(&start.stream);
        for layer in 0..end {
            eprintln!("full probe position={position} layer={layer} begin");
            if layer == shape.ple_layer {
                let mut residual = step.read_f32(state, start.stream.len()).unwrap();
                let update = feed.ple_update(token, &residual).unwrap();
                assert_eq!(residual.len(), update.len());
                for (value, delta) in residual.iter_mut().zip(update) {
                    *value += delta;
                }
                write(&residual);
            }
            if let Some(trace) = &mut trace {
                let mut ids = None;
                {
                    let moe = owner.bind(&bound).unwrap();
                    let mut tap = trace::RouteTap::new(moe, &mut ids);
                    step.run_layer(layer, position, 1, &mut tap).unwrap();
                }
                trace.record(
                    step,
                    layer,
                    ids.expect("router input and IDs captured inside MoE"),
                );
            } else if let Some(trace) = &mut stateful {
                let mut ids = None;
                {
                    let moe = owner.bind(&bound).unwrap();
                    let mut tap = trace::RouteTap::new(moe, &mut ids);
                    step.run_layer(layer, position, 1, &mut tap).unwrap();
                }
                trace.record(
                    step,
                    position,
                    token,
                    layer,
                    ids.expect("router input and IDs captured inside MoE"),
                );
            } else {
                step.run_layer(layer, position, 1, &mut owner.bind(&bound).unwrap())
                    .unwrap();
            }
            let health = step.health().unwrap();
            eprintln!("full probe position={position} layer={layer} end health={health:?}");
            assert_eq!(
                health,
                (0, 0),
                "unhealthy position={position} layer={layer}"
            );
        }
        if trace_first_token {
            step.finish_token().unwrap();
            trace.take().unwrap().finish(end);
            return;
        }
        if position + 1 == prompt.len() && end == layers {
            step.head(1).unwrap();
            let got = step
                .read_f32(step.slot(Slot::Logits).unwrap(), expected.len())
                .unwrap();
            assert_eq!(got.len(), expected.len());
            let actual_greedy = greedy(&got);
            let expected_greedy = greedy(&expected);
            let error = relative_l2(&got, &expected);
            eprintln!(
                "first logits count={} relative_l2={error:.9} gpu_greedy={actual_greedy} host_greedy={expected_greedy} gpu_max={} host_max={}",
                got.len(),
                got[actual_greedy],
                expected[expected_greedy]
            );
            if let Some(path) = std::env::var_os("FIFI_FLASH_GGUF_FULL_PROBE_LOGITS_OUT") {
                let output = std::fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(path)
                    .unwrap();
                serde_json::to_writer(output, &got).unwrap();
            }
            // Exploratory threshold inherited from the tail probe, not a validated tolerance.
            if !trace_stateful {
                assert!(
                    error.is_finite() && error < 0.05 && actual_greedy == expected_greedy,
                    "first logits mismatch: relative_l2={error}, gpu_greedy={actual_greedy}, host_greedy={expected_greedy}"
                );
            } else {
                assert!(
                    error.is_finite(),
                    "non-finite stateful diagnostic logits error"
                );
                eprintln!("stateful GPU logits comparison is diagnostic, not a parity assertion");
            }
        }
        step.finish_token().unwrap();
    }
    if let Some(trace) = stateful {
        trace.finish(prompt.len());
    }
}
