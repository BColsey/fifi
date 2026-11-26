//! Staged GGUF prefix probes. The actual residual crosses the existing host PLE
//! boundary; this is not production execution or a device-resident PLE.
#[path = "gguf_prefix_probe/delta_diagnostic.rs"]
mod delta_diagnostic;
#[path = "gguf_prefix_probe/delta_projection_probe.rs"]
mod delta_projection_probe;
#[path = "gguf_prefix_probe/layer5_diagnostic.rs"]
mod layer5_diagnostic;
#[path = "gguf_prefix_probe/qsa_diagnostic.rs"]
mod qsa_diagnostic;

use super::*;
use crate::family::qwen::flash_next::{
    cuda::{
        driver::TokenFeeds,
        staged_moe::{GgufMoeOwner, GgufStep},
    },
    load::gguf::{
        GgufModel,
        routed::PersistentRouted,
        routed_probe,
        staged_trunk::GgufTrunk,
        tests::{PinnedRouterReference, pinned_router_host_data, pinned_shared_host_data},
        token_feed::GgufTokenFeeds,
    },
};
use crate::kernels::cuda::{CudaContext, resident::step_params::StepParams};
use std::path::PathBuf;

enum MixerDiagnostic {
    Delta(delta_diagnostic::Reference),
    Qsa(qsa_diagnostic::Reference),
}

struct LayerReference {
    layer: usize,
    capture: PathBuf,
    input: Vec<f32>,
    shared: Vec<f32>,
    router: PinnedRouterReference,
    mixer: Option<MixerDiagnostic>,
}

fn upload_layer(
    ctx: &CudaContext,
    stream: &DeviceStream,
    model: &GgufModel,
    source: (usize, PathBuf),
) -> (Vec<CudaWeight>, LayerReference) {
    let (layer, capture) = source;
    let weights = model.upload_layer_trunk(layer, ctx, stream).unwrap();
    (weights, load_reference(model, (layer, capture)))
}

fn prompt_references(model: &GgufModel, captures: Vec<Vec<PathBuf>>) -> Vec<Vec<LayerReference>> {
    captures
        .into_iter()
        .map(|paths| {
            paths
                .into_iter()
                .enumerate()
                .map(|(layer, path)| load_reference(model, (layer, path)))
                .collect()
        })
        .collect()
}

fn load_reference(model: &GgufModel, source: (usize, PathBuf)) -> LayerReference {
    let (layer, capture) = source;
    let (input, mut shared_host) = pinned_shared_host_data(layer, &capture);
    let router = pinned_router_host_data(model, &input, layer);
    let shared = shared_host.pop().expect("captured shared expert output");
    let mixer = if std::env::var_os("FIFI_FLASH_GGUF_QSA_DIAG").is_none() {
        None
    } else if model.config().kinds()[layer] {
        Some(MixerDiagnostic::Qsa(qsa_diagnostic::Reference::load(
            layer, &capture,
        )))
    } else {
        Some(MixerDiagnostic::Delta(delta_diagnostic::Reference::load(
            layer, &capture,
        )))
    };
    LayerReference {
        layer,
        capture,
        input,
        shared,
        router,
        mixer,
    }
}

/// Probe the router with the captured host input, then restore the real GPU input and route.
fn diagnose_router_input(
    step: &FlashStep<'_>,
    reference: &LayerReference,
    actual: &[f32],
    exact: bool,
) {
    let mixed = step.slot(Slot::Mixed(Site::Mlp)).unwrap();
    let write = |values: &[f32]| {
        mixed
            .buffer()
            .write_at(
                &step.stream,
                mixed.offset(),
                &values
                    .iter()
                    .flat_map(|v| v.to_ne_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
    };
    write(&reference.input);
    let (_, teacher_match) =
        measure_composed_router(step, reference.layer, &reference.router, exact);
    eprintln!(
        "host-input router diagnostic: route_matches_host={teacher_match}; not a propagated execution"
    );
    write(actual);
    let (_, restored_match) =
        measure_composed_router(step, reference.layer, &reference.router, exact);
    assert!(
        !restored_match,
        "actual route mismatch did not reproduce after input restoration"
    );
    eprintln!("actual GPU input and route restored before aggregate diagnostic");
}

fn compare_persistent(
    step: &FlashStep<'_>,
    bank: &mut PersistentRouted<'_>,
    layer: usize,
    position: usize,
) {
    let bufs = step.moe_bufs().unwrap();
    let dims = step.moe_dims(1);
    let baseline = step.read_f32(bufs.route.resid_out, dims.hidden).unwrap();
    let started = std::time::Instant::now();
    let first = bank.run(layer, &bufs, &dims).unwrap();
    let actual = step.read_f32(bufs.route.resid_out, dims.hidden).unwrap();
    let first_ms = started.elapsed().as_secs_f64() * 1e3;
    assert!(
        actual == baseline,
        "persistent result differs from separately staged probe: rel L2={}",
        relative_l2(&actual, &baseline)
    );
    let started = std::time::Instant::now();
    let repeat = bank.run(layer, &bufs, &dims).unwrap();
    let replay = step.read_f32(bufs.route.resid_out, dims.hidden).unwrap();
    let replay_ms = started.elapsed().as_secs_f64() * 1e3;
    assert_eq!(repeat.ops.len(), 0, "immediate replay should be all hits");
    assert_eq!(repeat.hits, dims.top_k);
    assert!(replay == baseline, "hit replay changed the output");
    eprintln!(
        "persistent routed layer={layer} position={position} first_misses={} first_ms={first_ms:.3} replay_hits={} replay_ms={replay_ms:.3} element_equal=true (synchronous ID read, disk refill, expert kernels and output read; cache state uncontrolled)",
        first.ops.len(),
        repeat.hits
    );

    // Separate policy control: modify one route ID, then restore the actual device route.
    // It is not part of the propagated parity result above or below.
    if layer == 0 && position == 0 {
        let ids = bufs.route.ids;
        let mut raw = vec![0; dims.top_k * 4];
        ids.buffer()
            .read_at(&step.stream, ids.offset(), &mut raw)
            .unwrap();
        let original: Vec<i32> = raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| i32::from_ne_bytes(*b))
            .collect();
        let row = bank.slot_row(layer).to_vec();
        let filler = row.iter().position(|&slot| slot == 0).unwrap() as i32;
        assert!(
            !original.contains(&filler),
            "the eviction control needs a separate filler"
        );
        let rank = 0;
        let previous_slot = row[original[rank] as usize];
        let replacement = (0..dims.n_expert as i32)
            .find(|&id| row[id as usize] < 0)
            .unwrap();
        let mut changed = raw.clone();
        changed[rank * 4..rank * 4 + 4].copy_from_slice(&replacement.to_ne_bytes());
        ids.buffer()
            .write_at(&step.stream, ids.offset(), &changed)
            .unwrap();
        let evict = bank.run(layer, &bufs, &dims).unwrap();
        assert_eq!(evict.ops.len(), 1, "one synthetic replacement should miss");
        assert_eq!(
            i32::from(evict.ops[0].slot),
            previous_slot,
            "wrong expert evicted"
        );
        ids.buffer()
            .write_at(&step.stream, ids.offset(), &raw)
            .unwrap();
        let restore = bank.run(layer, &bufs, &dims).unwrap();
        assert_eq!(
            restore.ops.len(),
            1,
            "restoring the evicted expert should miss"
        );
        let output = step.read_f32(bufs.route.resid_out, dims.hidden).unwrap();
        assert!(
            output == baseline,
            "restored device route changed the output"
        );
        eprintln!(
            "persistent routed synthetic eviction: injected=1 restored=1; actual GPU route and output restored"
        );
    }
}

struct Run<'a, 'b> {
    position: usize,
    token: u32,
    misses: &'a [usize],
    persistent: Option<&'a mut PersistentRouted<'b>>,
    bridge: Option<(&'a mut GgufMoeOwner<'b>, &'a GgufStep<'b>)>,
}

fn run_layer<'a>(step: &FlashStep<'a>, reference: &LayerReference, host: &[f32], run: Run<'_, 'a>) {
    let Run {
        position,
        token,
        misses,
        persistent,
        bridge,
    } = run;
    let layer = reference.layer;
    eprintln!("four-layer prompt probe: position={position} token={token} layer={layer}");
    step.hyper_read(Site::Attn, &layer_name(layer, "attn_hyper_connection"), 1)
        .unwrap();
    match &reference.mixer {
        Some(MixerDiagnostic::Qsa(qsa)) => qsa_diagnostic::before(step, qsa, position, layer),
        Some(MixerDiagnostic::Delta(delta)) => {
            delta_diagnostic::before(step, delta, position, layer)
        }
        None => {}
    }
    step.mixer(layer, position, 1).unwrap();
    match &reference.mixer {
        Some(MixerDiagnostic::Qsa(qsa)) => qsa_diagnostic::after(step, qsa, position, layer),
        Some(MixerDiagnostic::Delta(delta)) => {
            delta_diagnostic::after(step, delta, position, layer)
        }
        None => {}
    }
    step.hyper_apply(Site::Attn, 1).unwrap();
    step.hyper_read(Site::Mlp, &layer_name(layer, "mlp_hyper_connection"), 1)
        .unwrap();
    let input = step
        .read_f32(
            step.slot(Slot::Mixed(Site::Mlp)).unwrap(),
            reference.input.len(),
        )
        .unwrap();
    let input_error = relative_l2(&input, &reference.input);
    eprintln!("four-layer FFN input relative L2={input_error:.9}");
    let exact = position == 0 && layer == 0;
    let (_, routes_match) = measure_composed_router(step, layer, &reference.router, exact);
    if !routes_match {
        diagnose_router_input(step, reference, &input, exact);
    }
    step.gguf_shared_expert(layer).unwrap();
    let shared = step
        .read_f32(step.slot(Slot::SharedOut).unwrap(), reference.shared.len())
        .unwrap();
    let shared_error = relative_l2(&shared, &reference.shared);
    let mut errors = (0.0, 0.0);
    for &initial_misses in misses {
        errors = routed_probe::run(
            &step.ctx,
            &step.stream,
            routed_probe::Inputs {
                layer,
                precision: step.gguf_decode_precision(),
                host_capture: &reference.capture,
                x: step.slot(Slot::Mixed(Site::Mlp)).unwrap(),
                route_ids: step.slot(Slot::TopkId).unwrap(),
                route_weights: step.slot(Slot::TopkWeight).unwrap(),
                initial_misses,
                diagnose_route_mismatch: !routes_match,
                shared_gate: step.slot(Slot::SharedGate).unwrap(),
                shared_out: step.slot(Slot::SharedOut).unwrap(),
                block_out: step.slot(Slot::BlockOut(Site::Mlp)).unwrap(),
            },
            &reference.router,
        );
        if routes_match {
            assert!(errors.0 < 0.08 && errors.1 < 0.08);
        } else {
            eprintln!(
                "route diagnostic only: aggregates include input, routing and numeric differences; strict parity remains failed"
            );
            break;
        }
    }
    assert!(routes_match, "GPU and host expert routes differ");
    if let Some(persistent) = persistent {
        compare_persistent(step, persistent, layer, position);
    }
    if let Some((bridge, bound_step)) = bridge {
        let output = step.slot(Slot::BlockOut(Site::Mlp)).unwrap();
        let expected = step.read_f32(output, step.geometry.hidden).unwrap();
        // The prior output is not evidence that the bridge wrote every element. Poison it,
        // after checking that the subsequent block's live input, gate and stream are disjoint.
        for slot in [Slot::Mixed(Site::Mlp), Slot::Gate(Site::Mlp), Slot::Stream] {
            let live = step.slot(slot).unwrap();
            let overlaps = std::ptr::eq(output.buffer(), live.buffer())
                && output.offset() < live.offset() + live.bytes()
                && live.offset() < output.offset() + output.bytes();
            assert!(
                !overlaps,
                "cannot poison bridge output overlapping {slot:?}"
            );
        }
        output
            .buffer()
            .write_at(
                &step.stream,
                output.offset(),
                &f32::NAN.to_ne_bytes().repeat(step.geometry.hidden),
            )
            .unwrap();
        step.moe_block(layer, position, &mut bridge.bind(bound_step).unwrap(), 1)
            .unwrap();
        let actual = step.read_f32(output, step.geometry.hidden).unwrap();
        assert!(expected.iter().all(|value| value.is_finite()));
        assert!(
            actual
                .iter()
                .zip(&expected)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "GGUF bridge output differs from independent staged probe: rel L2={}",
            relative_l2(&actual, &expected)
        );
        eprintln!("GGUF MoeBlock bridge layer={layer} position={position} bit_exact=true");
    }
    step.hyper_apply(Site::Mlp, 1).unwrap();
    let residual = step
        .read_f32(step.slot(Slot::Stream).unwrap(), host.len())
        .unwrap();
    let residual_error = relative_l2(&residual, host);
    eprintln!(
        "four-layer prompt probe: position={position} token={token} layer={layer} ffn_input={input_error:.6} shared={shared_error:.6} routed={:.6} combined={:.6} residual={residual_error:.6}",
        errors.0, errors.1
    );
    assert!(input_error < 0.05 && shared_error < 0.08 && residual_error < 0.05);
}

/// Isolated host-input check; this is not a propagated FlashStep or a parity qualification.
#[test]
#[ignore = "requires the pinned GGUF, layer-zero host capture and an admitted CUDA device"]
fn pinned_matched_layer_zero_moe_diagnostic() {
    assert!(
        std::env::var_os("FIFI_FLASH_GGUF_MATCHED_EXPERT_DOWN").is_some(),
        "enable matched expert diagnostic"
    );
    assert!(
        std::env::var_os("FIFI_FLASH_GGUF_MATCHED_EXPERT_DOWN_OUT").is_some(),
        "set a new output receipt path"
    );
    let path = std::env::var_os("FIFI_FLASH_GGUF").expect("verified first shard");
    let capture =
        std::env::var_os("FIFI_FLASH_GGUF_HOST_CAPTURE").expect("verified layer-zero host capture");
    assert_pinned_file(
        &path,
        "9d5644b371ffac921427c8021cc7242dab8e3ba37d7b48d1e79fe3770fbf9915",
    );
    assert_pinned_file(
        &capture,
        "b0873cedfb9733049335904d64635d20b155e68614de0166a287333f5b5640e6",
    );
    let model = GgufModel::open(Path::new(&path)).unwrap().unwrap();
    let capture = PathBuf::from(capture);
    let (input, shared) = pinned_shared_host_data(0, &capture);
    let router = pinned_router_host_data(&model, &input, 0);
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let weights =
        CudaWeights::from_native(model.upload_layer_trunk(0, &ctx, &stream).unwrap()).unwrap();
    let step = FlashStep::new_gguf_decode(
        &ctx,
        &stream,
        &weights,
        model.config().model_shape().unwrap(),
        model.config().kinds(),
        1,
        model.map().max_quantized_input_width().unwrap(),
        0..1,
    )
    .unwrap();
    step.begin_token().unwrap();
    let mixed = step.slot(Slot::Mixed(Site::Mlp)).unwrap();
    mixed
        .buffer()
        .write_at(
            &stream,
            mixed.offset(),
            &input
                .iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let (_, matched) = measure_composed_router(&step, 0, &router, true);
    assert!(matched, "host-input device routes must match in rank order");
    step.gguf_shared_expert(0).unwrap();
    let width = shared[0].len();
    let gu = step
        .read_f32(step.slot(Slot::SharedGu).unwrap(), 2 * width)
        .unwrap();
    let act = step
        .read_f32(step.slot(Slot::SharedAct).unwrap(), width)
        .unwrap();
    let out = step
        .read_f32(step.slot(Slot::SharedOut).unwrap(), input.len())
        .unwrap();
    let errors = [
        relative_l2(&gu[..width], &shared[0]),
        relative_l2(&gu[width..], &shared[1]),
        relative_l2(&act, &shared[2]),
        relative_l2(&out, &shared[3]),
    ];
    eprintln!(
        "matched shared expert: gate={:.9} up={:.9} act={:.9} down={:.9}",
        errors[0], errors[1], errors[2], errors[3]
    );
    assert!(errors.iter().all(|&v| v.is_finite() && v < 0.05));
    let errors = routed_probe::run(
        &ctx,
        &stream,
        routed_probe::Inputs {
            layer: 0,
            precision: step.gguf_decode_precision(),
            host_capture: &capture,
            x: mixed,
            route_ids: step.slot(Slot::TopkId).unwrap(),
            route_weights: step.slot(Slot::TopkWeight).unwrap(),
            initial_misses: 0,
            diagnose_route_mismatch: false,
            shared_gate: step.slot(Slot::SharedGate).unwrap(),
            shared_out: step.slot(Slot::SharedOut).unwrap(),
            block_out: step.slot(Slot::BlockOut(Site::Mlp)).unwrap(),
        },
        &router,
    );
    assert!(errors.0 < 0.05 && errors.1 < 0.05);
    assert_eq!(step.health().unwrap(), (0, 0));
    eprintln!(
        "matched layer-zero MoE diagnostics complete; not propagated execution or model parity"
    );
}

fn captures() -> Vec<PathBuf> {
    let inputs = [
        (
            "FIFI_FLASH_GGUF_HOST_CAPTURE",
            "b0873cedfb9733049335904d64635d20b155e68614de0166a287333f5b5640e6",
        ),
        (
            "FIFI_FLASH_GGUF_HOST_CAPTURE_LAYER1",
            "32f7f3ac5de6366f4f5b14b825963eb5ca1332790d1727d8bdd5c13b63264e35",
        ),
        (
            "FIFI_FLASH_GGUF_HOST_CAPTURE_LAYER2",
            "94d05235bd0c42e9c62760c63a4d5f4cd964d402720e0fa81bfaa6251920f403",
        ),
        (
            "FIFI_FLASH_GGUF_HOST_CAPTURE_LAYER3",
            "4778240e7c7cf92230b1fcf2d5fb84a6846bcd89727001aa756a8873db3fa1b8",
        ),
    ];
    inputs
        .into_iter()
        .map(|(key, digest)| {
            let path = std::env::var_os(key).expect("set verified layer capture");
            assert_pinned_file(&path, digest);
            PathBuf::from(path)
        })
        .collect()
}

#[test]
#[ignore = "requires the pinned GGUF, four host captures and an admitted CUDA device"]
fn four_layers_with_host_ple_match_pinned_host() {
    let path = std::env::var_os("FIFI_FLASH_GGUF").expect("set verified checkpoint first shard");
    let layers = std::env::var_os("FIFI_FLASH_GGUF_HOST_LAYERS").expect("set verified host trace");
    // First shard and host captures are verified here. All remaining checkpoint shards
    // still require the publisher full-file checks before running this test.
    assert_pinned_file(
        &path,
        "9d5644b371ffac921427c8021cc7242dab8e3ba37d7b48d1e79fe3770fbf9915",
    );
    assert_pinned_file(
        &layers,
        "62fdd25ecb4f3e14f2f4417de026421e6e6557c972eadee3e779e9cc6656bace",
    );
    let captures = captures();
    let host: Vec<Vec<Vec<f32>>> = serde_json::from_slice(&std::fs::read(layers).unwrap()).unwrap();
    let model = GgufModel::open(Path::new(&path)).unwrap().unwrap();
    let shape = model.config().model_shape().unwrap();
    assert_eq!(shape.ple_layer, 1);
    assert_eq!(&model.config().kinds()[..4], &[false, false, false, true]);
    let ngram = model.ngram().unwrap();
    let mut feed = GgufTokenFeeds::new(&model, &ngram, 8).unwrap();
    let start = feed.begin(9707).unwrap();
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let mut weights = Vec::new();
    let mut references = Vec::new();
    for (layer, capture) in captures.into_iter().enumerate() {
        let (layer_weights, reference) = upload_layer(&ctx, &stream, &model, (layer, capture));
        weights.extend(layer_weights);
        references.push(reference);
    }
    let weights = CudaWeights::from_native(weights).unwrap();
    let max_k = model.map().max_quantized_input_width().unwrap();
    let step = FlashStep::new_gguf_decode(
        &ctx,
        &stream,
        &weights,
        shape,
        model.config().kinds(),
        1,
        max_k,
        0..4,
    )
    .unwrap();
    eprintln!(
        "four-layer staged device live bytes={} max native input={max_k}",
        ctx.live_bytes()
    );
    step.begin_token().unwrap();
    let (words, at) = step.words_buffer().unwrap();
    words
        .write_at(&stream, at, &StepParams::new(1, 0).unwrap().to_bytes())
        .unwrap();
    let state = step.slot(Slot::Stream).unwrap();
    state
        .buffer()
        .write_at(
            &stream,
            state.offset(),
            &start
                .stream
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    for reference in &references {
        if reference.layer == shape.ple_layer {
            let mut residual = step.read_f32(state, start.stream.len()).unwrap();
            let update = feed.ple_update(9707, &residual).unwrap();
            assert_eq!(residual.len(), update.len());
            for (value, delta) in residual.iter_mut().zip(update) {
                *value += delta;
            }
            state
                .buffer()
                .write_at(
                    &stream,
                    state.offset(),
                    &residual
                        .iter()
                        .flat_map(|v| v.to_ne_bytes())
                        .collect::<Vec<_>>(),
                )
                .unwrap();
        }
        let misses: &[usize] = if reference.layer == 3 {
            &[0, 2, 10]
        } else {
            &[0]
        };
        run_layer(
            &step,
            reference,
            &host[0][reference.layer],
            Run {
                position: 0,
                token: 9707,
                misses,
                persistent: None,
                bridge: None,
            },
        );
    }
    assert_eq!(step.health().unwrap(), (0, 0));
}

/// A manifest binds every position/layer fixture and the residuals to one host run.
fn prompt_captures() -> (Vec<Vec<PathBuf>>, PathBuf) {
    let dir = PathBuf::from(
        std::env::var_os("FIFI_FLASH_GGUF_PROMPT_CAPTURE_DIR").expect("set capture directory"),
    );
    let manifest_path = dir.join("manifest.json");
    assert_pinned_file(
        manifest_path.as_os_str(),
        "880ef30a997107dd37dbfd2cc40c99dbcf80676069333ba542d111f80ee2f416",
    );
    let manifest: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(manifest_path).unwrap()).unwrap();
    assert_eq!(manifest["tokens"], serde_json::json!([9707, 11, 1879]));
    assert_eq!(manifest["layers"], 4);
    let layers = PathBuf::from(
        std::env::var_os("FIFI_FLASH_GGUF_PROMPT_LAYERS").expect("set prompt residuals"),
    );
    assert_pinned_file(
        layers.as_os_str(),
        manifest["prefix_layers_sha256"].as_str().unwrap(),
    );
    let files = manifest["files"].as_array().unwrap();
    assert_eq!(files.len(), 12);
    let mut captures = vec![vec![PathBuf::new(); 4]; 3];
    for entry in files {
        let position = entry["position"].as_u64().unwrap() as usize;
        let layer = entry["layer"].as_u64().unwrap() as usize;
        assert!(position < 3 && layer < 4 && captures[position][layer].as_os_str().is_empty());
        let name = format!("position-{position}-layer-{layer}.jsonl");
        assert_eq!(entry["path"].as_str().unwrap(), name);
        let path = dir.join(name);
        assert_pinned_file(path.as_os_str(), entry["sha256"].as_str().unwrap());
        captures[position][layer] = path;
    }
    assert!(
        captures
            .iter()
            .flatten()
            .all(|path| !path.as_os_str().is_empty())
    );
    (captures, layers)
}

#[test]
#[ignore = "requires pinned GGUF, three-position host fixtures and an admitted CUDA device"]
fn three_prompt_tokens_with_host_ple_match_pinned_host() {
    let path = std::env::var_os("FIFI_FLASH_GGUF").expect("set verified checkpoint first shard");
    assert_pinned_file(
        &path,
        "9d5644b371ffac921427c8021cc7242dab8e3ba37d7b48d1e79fe3770fbf9915",
    );
    let (captures, layers) = prompt_captures();
    let host: Vec<Vec<Vec<f32>>> =
        serde_json::from_reader(std::fs::File::open(layers).unwrap()).unwrap();
    assert_eq!(host.len(), 3);
    assert!(host.iter().all(|layers| layers.len() == 4));
    let model = GgufModel::open(Path::new(&path)).unwrap().unwrap();
    let shape = model.config().model_shape().unwrap();
    assert_eq!(shape.ple_layer, 1);
    assert_eq!(&model.config().kinds()[..4], &[false, false, false, true]);
    let ngram = model.ngram().unwrap();
    let mut feed = GgufTokenFeeds::new(&model, &ngram, 8).unwrap();
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let trunk = GgufTrunk::upload(&model, &ctx, &stream, 0..4).unwrap();
    let references = prompt_references(&model, captures);
    let max_k = model.map().max_quantized_input_width().unwrap();
    let bound_step = GgufStep::new(&ctx, &stream, &trunk, 3).unwrap();
    let step = bound_step.step();
    eprintln!(
        "three-token persistent device bytes before routed banks={} max native input={max_k}",
        ctx.live_bytes()
    );
    let mut persistent = if std::env::var_os("FIFI_FLASH_GGUF_TEST_PERSISTENT_ROUTED").is_some() {
        Some(PersistentRouted::new(&ctx, &stream, &model, 0..4, shape.moe.top_k() + 1).unwrap())
    } else {
        None
    };
    let mut bridge = if std::env::var_os("FIFI_FLASH_GGUF_TEST_MOE_BRIDGE").is_some() {
        assert!(persistent.is_none(), "select only one staged bank probe");
        Some(GgufMoeOwner::new(&bound_step, shape.moe.top_k() + 1).unwrap())
    } else {
        None
    };
    for (position, token) in [9707u32, 11, 1879].into_iter().enumerate() {
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
        state
            .buffer()
            .write_at(
                &stream,
                state.offset(),
                &start
                    .stream
                    .iter()
                    .flat_map(|v| v.to_ne_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        for reference in &references[position] {
            if reference.layer == shape.ple_layer {
                let mut residual = step.read_f32(state, start.stream.len()).unwrap();
                let update = feed.ple_update(token, &residual).unwrap();
                assert_eq!(residual.len(), update.len());
                for (value, delta) in residual.iter_mut().zip(update) {
                    *value += delta;
                }
                state
                    .buffer()
                    .write_at(
                        &stream,
                        state.offset(),
                        &residual
                            .iter()
                            .flat_map(|v| v.to_ne_bytes())
                            .collect::<Vec<_>>(),
                    )
                    .unwrap();
            }
            let misses: &[usize] = if position == 2 && reference.layer == 3 {
                &[0, 2, 10]
            } else {
                &[0]
            };
            run_layer(
                step,
                reference,
                &host[position][reference.layer],
                Run {
                    position,
                    token,
                    misses,
                    persistent: persistent.as_mut(),
                    bridge: bridge.as_mut().map(|owner| (owner, &bound_step)),
                },
            );
        }
        step.finish_token().unwrap();
    }
    if let Some(mut bank) = persistent {
        eprintln!("persistent routed final stats: {:?}", bank.stats());
        let bufs = step.moe_bufs().unwrap();
        bufs.route
            .ids
            .buffer()
            .write_at(
                &stream,
                bufs.route.ids.offset(),
                &(-1i32).to_ne_bytes().repeat(shape.moe.top_k()),
            )
            .unwrap();
        assert!(bank.run(0, &bufs, &step.moe_dims(1)).is_err());
        assert!(
            bank.run(0, &bufs, &step.moe_dims(1)).is_err(),
            "failed routed owner must remain poisoned"
        );
    }
}

/// Exercises the actual layer entry point. PLE remains host-side and the F32 DeltaNet output
/// is an explicit diagnostic, so this is not full-model or default-path qualification.
#[test]
#[ignore = "requires pinned GGUF, prompt fixtures and an admitted CUDA device"]
fn gguf_moe_bridge_runs_three_stateful_tokens_through_layer_entry() {
    assert!(
        std::env::var_os("FIFI_FLASH_GGUF_TEST_F32_DELTA_OUT").is_some(),
        "this pinned diagnostic requires the F32 DeltaNet output experiment"
    );
    let path = std::env::var_os("FIFI_FLASH_GGUF").expect("set verified checkpoint first shard");
    assert_pinned_file(
        &path,
        "9d5644b371ffac921427c8021cc7242dab8e3ba37d7b48d1e79fe3770fbf9915",
    );
    let (captures, layers) = prompt_captures();
    let host: Vec<Vec<Vec<f32>>> =
        serde_json::from_reader(std::fs::File::open(layers).unwrap()).unwrap();
    assert_eq!(host.len(), 3);
    assert!(host.iter().all(|layers| layers.len() == 4));
    let model = GgufModel::open(Path::new(&path)).unwrap().unwrap();
    let shape = model.config().model_shape().unwrap();
    assert_eq!(shape.ple_layer, 1);
    assert_eq!(&model.config().kinds()[..4], &[false, false, false, true]);
    let ngram = model.ngram().unwrap();
    let mut feed = GgufTokenFeeds::new(&model, &ngram, 8).unwrap();
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let trunk = GgufTrunk::upload(&model, &ctx, &stream, 0..4).unwrap();
    let references = prompt_references(&model, captures);
    let bound_step = GgufStep::new(&ctx, &stream, &trunk, 3).unwrap();
    let step = bound_step.step();
    let mut bridge = GgufMoeOwner::new(&bound_step, shape.moe.top_k() + 1).unwrap();
    for (position, token) in [9707u32, 11, 1879].into_iter().enumerate() {
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
        state
            .buffer()
            .write_at(
                &stream,
                state.offset(),
                &start
                    .stream
                    .iter()
                    .flat_map(|v| v.to_ne_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        for reference in &references[position] {
            let layer = reference.layer;
            if layer == shape.ple_layer {
                let mut residual = step.read_f32(state, start.stream.len()).unwrap();
                let update = feed.ple_update(token, &residual).unwrap();
                assert_eq!(residual.len(), update.len());
                for (value, delta) in residual.iter_mut().zip(update) {
                    *value += delta;
                }
                state
                    .buffer()
                    .write_at(
                        &stream,
                        state.offset(),
                        &residual
                            .iter()
                            .flat_map(|v| v.to_ne_bytes())
                            .collect::<Vec<_>>(),
                    )
                    .unwrap();
            }
            step.run_layer(layer, position, 1, &mut bridge.bind(&bound_step).unwrap())
                .unwrap();
            let ids = step.slot(Slot::TopkId).unwrap();
            let mut raw = vec![0; shape.moe.top_k() * 4];
            ids.buffer()
                .read_at(&stream, ids.offset(), &mut raw)
                .unwrap();
            let mut actual: Vec<i32> = raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| i32::from_ne_bytes(*b))
                .collect();
            let mut expected: Vec<i32> = reference
                .router
                .route
                .iter()
                .map(|(id, _)| *id as i32)
                .collect();
            actual.sort_unstable();
            expected.sort_unstable();
            assert_eq!(
                actual, expected,
                "GPU and host route sets differ at position {position}, layer {layer}"
            );
            let residual = step.read_f32(state, host[position][layer].len()).unwrap();
            let error = relative_l2(&residual, &host[position][layer]);
            eprintln!(
                "GGUF run_layer bridge position={position} layer={layer} route_set_match=true residual_l2={error:.9}"
            );
            assert!(error < 0.05);
            assert_eq!(step.health().unwrap(), (0, 0));
        }
        step.finish_token().unwrap();
    }
    let wrong_stream = ctx.stream().unwrap();
    let wrong = Launch {
        ctx: &ctx,
        stream: &wrong_stream,
    };
    let correct = Launch {
        ctx: &ctx,
        stream: &stream,
    };
    let bufs = step.moe_bufs().unwrap();
    let dims = step.moe_dims(1);
    let mut bound = bridge.bind(&bound_step).unwrap();
    let failure = bound.run(&wrong, 0, 0, &bufs, &dims).unwrap_err();
    assert!(
        matches!(&failure, EngError::BadRequest(message) if message.contains("does not belong")),
        "wrong-stream failure: {failure:?}"
    );
    let poisoned = bound.run(&correct, 0, 0, &bufs, &dims).unwrap_err();
    assert!(
        matches!(&poisoned, EngError::BadRequest(message) if message.contains("poisoned")),
        "subsequent failure: {poisoned:?}"
    );
    eprintln!("GGUF MoeBlock bridge foreign stream rejected and owner poisoned");
}
