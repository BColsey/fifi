//! Pinned second-token DeltaNet boundaries; fixture layer numbers are not model dimensions.
#[path = "delta_history.rs"]
mod delta_history;

use super::*;

struct Fixture {
    layer: usize,
    trace_var: &'static str,
    trace_sha: &'static str,
    host0_var: &'static str,
    host0_sha: &'static str,
    host1_var: &'static str,
    host1_sha: &'static str,
    qsa_input_f32: bool,
}

const LAYER_FIVE: Fixture = Fixture {
    layer: 5,
    trace_var: "FIFI_FLASH_GGUF_LAYER5_GPU_TRACE",
    trace_sha: "e289735857931030d317c5c8bcfa9c444e49b5da9be0fed3f841ec8292d7e56b",
    host0_var: "FIFI_FLASH_GGUF_LAYER5_POSITION0_CAPTURE",
    host0_sha: "eafcc69c9e8399258353c364b354a0d710417d559238b4f41ca50304e79b84b4",
    host1_var: "FIFI_FLASH_GGUF_LAYER5_POSITION1_CAPTURE",
    host1_sha: "280b33d63703d79682a4ffd4a28f45062bef71d6852367906c4220f9f89d0082",
    qsa_input_f32: false,
};
const LAYER_FOUR: Fixture = Fixture {
    layer: 4,
    trace_var: "FIFI_FLASH_GGUF_LAYER4_GPU_TRACE",
    trace_sha: "1fa333a5139985673c073dba0981d3926986a534fb70b7351faaa2e107071b69",
    host0_var: "FIFI_FLASH_GGUF_LAYER4_POSITION0_CAPTURE",
    host0_sha: "e3b71600e9edf317065402d406eaa0d976916c4c9d3ff380ed16050cb5770ced",
    host1_var: "FIFI_FLASH_GGUF_LAYER4_POSITION1_CAPTURE",
    host1_sha: "e9dd69ee86cf5f39c589217ed141f3c3970435b178a70d5e9ed69abd2bfa8928",
    qsa_input_f32: true,
};

fn pinned_path(variable: &str, hash: &str) -> PathBuf {
    let path = PathBuf::from(std::env::var_os(variable).expect(variable));
    assert_pinned_file(path.as_os_str(), hash);
    path
}

fn trace_row(rows: &[serde_json::Value], layer: usize) -> &serde_json::Value {
    let mut found = rows
        .iter()
        .filter(|row| row["position"] == 1 && row["token"] == 11 && row["layer"] == layer);
    let row = found.next().expect("trace layer missing");
    assert!(found.next().is_none(), "duplicate trace layer");
    row
}

fn pinned_trace(fixture: &Fixture) -> Vec<serde_json::Value> {
    let path = pinned_path(fixture.trace_var, fixture.trace_sha);
    let rows: Vec<serde_json::Value> =
        serde_json::from_reader(std::fs::File::open(path).unwrap()).unwrap();
    assert_eq!(rows.len(), 144);
    trace_row(&rows, fixture.layer - 1);
    trace_row(&rows, fixture.layer);
    rows
}

fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn f32_row(row: &serde_json::Value, field: &str) -> Vec<f32> {
    serde_json::from_value(row[field].clone()).unwrap()
}

fn measure_boundary(
    step: &FlashStep<'_>,
    reference: &LayerReference,
    trace: &[serde_json::Value],
    layer: usize,
) -> delta_history::MixerTrace {
    let previous = trace_row(trace, layer - 1);
    let target = trace_row(trace, layer);
    let state = step.slot(Slot::Stream).unwrap();
    let expected_state = f32_row(previous, "gpu_stream");
    assert!(
        same_bits(
            &step.read_f32(state, expected_state.len()).unwrap(),
            &expected_state
        ),
        "preceding GPU state diverged from full trace"
    );
    step.hyper_read(Site::Attn, &layer_name(layer, "attn_hyper_connection"), 1)
        .unwrap();
    let Some(MixerDiagnostic::Delta(delta)) = &reference.mixer else {
        panic!("Delta diagnostic must be enabled")
    };
    delta_diagnostic::before(step, delta, 1, layer);
    let mixer_input = step
        .read_f32(
            step.slot(Slot::Mixed(Site::Attn)).unwrap(),
            step.geometry.hidden,
        )
        .unwrap();
    step.mixer(layer, 1, 1).unwrap();
    let mixer_core = step
        .read_f32(
            step.slot(Slot::GgufDeltaTile).unwrap(),
            step.geometry.delta.value_width(),
        )
        .unwrap();
    let mixer_output = step
        .read_f32(
            step.slot(Slot::BlockOut(Site::Attn)).unwrap(),
            step.geometry.hidden,
        )
        .unwrap();
    delta_diagnostic::after(step, delta, 1, layer);
    step.hyper_apply(Site::Attn, 1).unwrap();
    step.hyper_read(Site::Mlp, &layer_name(layer, "mlp_hyper_connection"), 1)
        .unwrap();
    let mixed = step.slot(Slot::Mixed(Site::Mlp)).unwrap();
    let actual = step.read_f32(mixed, reference.input.len()).unwrap();
    let expected = f32_row(target, "gpu_router_input");
    assert!(
        same_bits(&actual, &expected),
        "GPU router input diverged from full trace"
    );
    eprintln!(
        "layer-{layer} propagated FFN input relative_l2={:.9}",
        relative_l2(&actual, &reference.input)
    );
    let (_, matches) = measure_composed_router(step, layer, &reference.router, false);
    assert!(!matches, "expected route mismatch not reproduced");
    let ids = step.slot(Slot::TopkId).unwrap();
    let mut bytes = vec![0; step.geometry.top_k * 4];
    ids.buffer()
        .read_at(step.stream, ids.offset(), &mut bytes)
        .unwrap();
    let actual_ids: Vec<i32> = bytes
        .chunks_exact(4)
        .map(|x| i32::from_ne_bytes(x.try_into().unwrap()))
        .collect();
    let expected_ids: Vec<i32> = serde_json::from_value(target["gpu_ids"].clone()).unwrap();
    assert_eq!(
        actual_ids, expected_ids,
        "GPU route IDs diverged from full trace"
    );
    diagnose_router_input(step, reference, &actual, false);
    assert_eq!(step.health().unwrap(), (0, 0));
    step.finish_token().unwrap();
    eprintln!(
        "layer-{layer} propagated boundary reproduced full trace state, router input and IDs; diagnostic only"
    );
    delta_history::MixerTrace {
        input: mixer_input,
        core: mixer_core,
        output: mixer_output,
    }
}

fn run_fixture(fixture: &Fixture) {
    for variable in [
        "FIFI_FLASH_GGUF_TEST_F32_DELTA_IN",
        "FIFI_FLASH_GGUF_TEST_F32_DELTA_OUT",
        "FIFI_FLASH_GGUF_TEST_F32_QSA_OUT",
        "FIFI_FLASH_GGUF_QSA_DIAG",
    ] {
        assert!(
            std::env::var_os(variable).is_some(),
            "{variable} must be enabled"
        );
    }
    assert_eq!(
        std::env::var_os("FIFI_FLASH_GGUF_TEST_F32_QSA_IN").is_some(),
        fixture.qsa_input_f32,
        "QSA input precision must match the pinned trace"
    );
    for variable in [
        "FIFI_FLASH_GGUF_TEST_F16_DELTA_OUT",
        "FIFI_FLASH_GGUF_TEST_IQ4_F16_DOWN",
        "FIFI_FLASH_GGUF_TEST_SHARED_PRECISE",
        "FIFI_FLASH_GGUF_TEST_DELTA_BENCH",
    ] {
        assert!(
            std::env::var_os(variable).is_none(),
            "{variable} changes the pinned trajectory"
        );
    }
    let path = pinned_path(
        "FIFI_FLASH_GGUF",
        "9d5644b371ffac921427c8021cc7242dab8e3ba37d7b48d1e79fe3770fbf9915",
    );
    let trace = pinned_trace(fixture);
    let model = GgufModel::open(&path).unwrap().unwrap();
    let layer = fixture.layer;
    assert!(
        !model.config().kinds()[layer],
        "fixture layer must be DeltaNet"
    );
    let reference = load_reference(
        &model,
        (layer, pinned_path(fixture.host1_var, fixture.host1_sha)),
    );
    let shape = model.config().model_shape().unwrap();
    assert!(
        shape.ple_layer < layer,
        "PLE must precede the manually stepped layer"
    );
    let ngram = model.ngram().unwrap();
    let mut feed = GgufTokenFeeds::new(&model, &ngram, 3).unwrap();
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let trunk = GgufTrunk::upload(&model, &ctx, &stream, 0..layer + 1).unwrap();
    let bound = GgufStep::new(&ctx, &stream, &trunk, 3).unwrap();
    let step = bound.step();
    let mut owner = GgufMoeOwner::new(&bound, shape.moe.top_k() + 1).unwrap();
    for (position, token) in [9707u32, 11].into_iter().enumerate() {
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
        for current in 0..if position == 0 { layer + 1 } else { layer } {
            if current == shape.ple_layer {
                let mut residual = step.read_f32(state, start.stream.len()).unwrap();
                let update = feed.ple_update(token, &residual).unwrap();
                assert_eq!(residual.len(), update.len());
                for (value, delta) in residual.iter_mut().zip(update) {
                    *value += delta;
                }
                write(&residual);
            }
            step.run_layer(current, position, 1, &mut owner.bind(&bound).unwrap())
                .unwrap();
            assert_eq!(step.health().unwrap(), (0, 0));
        }
        if position == 0 {
            step.finish_token().unwrap();
        } else {
            let history = std::env::var_os("FIFI_FLASH_GGUF_DELTA_HISTORY_OUT")
                .map(|_| delta_history::History::read(step, layer));
            let live = measure_boundary(step, &reference, &trace, layer);
            if let Some(history) = &history {
                let Some(MixerDiagnostic::Delta(host1)) = &reference.mixer else {
                    unreachable!()
                };
                let host0 = delta_diagnostic::Reference::load(
                    layer,
                    &pinned_path(fixture.host0_var, fixture.host0_sha),
                );
                delta_history::run(delta_history::Inputs {
                    model: &model,
                    live_step: step,
                    live: &live,
                    gpu_history: history,
                    layer,
                    host0: &host0,
                    host1,
                });
            }
        }
    }
}

#[test]
#[ignore = "requires pinned GGUF, layer-5 host/GPU trace fixtures and an admitted CUDA device"]
fn pinned_second_token_layer_five_boundary() {
    run_fixture(&LAYER_FIVE);
}

#[test]
#[ignore = "requires pinned GGUF, layer-4 host/GPU trace fixtures and an admitted CUDA device"]
fn pinned_second_token_layer_four_boundary() {
    run_fixture(&LAYER_FOUR);
}
