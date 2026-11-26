//! Test-only propagated residual intervention; rows record live values before any substitution.
use super::*;

const END: usize = 5; // Includes the fixture's first divergent layer, not model geometry.
const PROMPT: [u32; 3] = [9707, 11, 1879];

fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn parse_mode(value: &str) -> Option<u8> {
    value.parse::<u8>().ok().filter(|&mask| mask <= 3)
}

fn selected(mask: u8, position: usize) -> bool {
    position < 2 && mask & (1 << position) != 0
}

fn mode() -> u8 {
    parse_mode(
        &std::env::var("FIFI_FLASH_GGUF_INTERVENTION_MASK").expect("explicit intervention mask"),
    )
    .expect("mask must be an integer selecting positions 0 and/or 1")
}

#[test]
fn intervention_mask_schedule_and_bitwise_comparison() {
    for (mask, expected) in [
        (0, [false, false, false]),
        (1, [true, false, false]),
        (2, [false, true, false]),
        (3, [true, true, false]),
    ] {
        assert_eq!(parse_mode(&mask.to_string()), Some(mask));
        assert_eq!(
            std::array::from_fn::<_, 3, _>(|pos| selected(mask, pos)),
            expected
        );
    }
    for value in ["", "-1", "4", "256", "1.0", "foo"] {
        assert_eq!(parse_mode(value), None);
    }
    assert!(same_bits(&[0.0, -0.0], &[0.0, -0.0]));
    assert!(!same_bits(&[0.0], &[-0.0]));
    assert!(!same_bits(&[1.0], &[1.0, 1.0]));
}

fn pinned_host(width: usize, layers: usize) -> Vec<Vec<Vec<f32>>> {
    let path = std::env::var_os("FIFI_FLASH_GGUF_STATEFUL_HOST_LAYERS")
        .expect("pinned uninterrupted host residual trace");
    assert_pinned_file(
        &path,
        "ca258df303d22dd6e804ed362728570a12a13e1308418abdf50cc76d86268b39",
    );
    let host: Vec<Vec<Vec<f32>>> =
        serde_json::from_reader(std::fs::File::open(path).unwrap()).unwrap();
    assert_eq!(host.len(), PROMPT.len());
    assert!(host.iter().all(|p| {
        p.len() == layers
            && p.iter()
                .all(|r| r.len() == width && r.iter().all(|x| x.is_finite()))
    }));
    host
}

fn replace_stream(step: &FlashStep<'_>, reference: &[f32], position: usize) {
    let state = step.slot(Slot::Stream).unwrap();
    let before = step.read_f32(state, reference.len()).unwrap();
    assert!(
        !same_bits(&before, reference),
        "intervention must change live residual"
    );
    let conv = step.read_state_bytes(StateSlot::DeltaConv(0)).unwrap();
    let rec = step.read_state_bytes(StateSlot::DeltaRec(0)).unwrap();
    state
        .buffer()
        .write_at(
            &step.stream,
            state.offset(),
            &reference
                .iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    assert!(
        same_bits(&step.read_f32(state, reference.len()).unwrap(), reference),
        "injected host residual must round-trip bitwise"
    );
    assert_eq!(
        step.read_state_bytes(StateSlot::DeltaConv(0)).unwrap(),
        conv
    );
    assert_eq!(step.read_state_bytes(StateSlot::DeltaRec(0)).unwrap(), rec);
    eprintln!(
        "layer-0 propagated intervention: position={position} pre_injection_relative_l2={:.9}; injected host residual bitwise, layer-0 device history unchanged",
        relative_l2(&before, reference)
    );
}

#[test]
#[ignore = "requires pinned GGUF, host residual trace and an admitted CUDA device"]
fn pinned_layer_zero_residual_intervention() {
    let mask = mode();
    for variable in [
        "FIFI_FLASH_GGUF_TEST_F32_DELTA_IN",
        "FIFI_FLASH_GGUF_TEST_F32_DELTA_OUT",
        "FIFI_FLASH_GGUF_TEST_F32_QSA_OUT",
        "FIFI_FLASH_GGUF_TEST_F32_QSA_IN",
    ] {
        assert!(
            std::env::var_os(variable).is_some(),
            "{variable} must match the pinned baseline"
        );
    }
    for variable in [
        "FIFI_FLASH_GGUF_TEST_SHARED_PRECISE",
        "FIFI_FLASH_GGUF_TEST_IQ4_F16_DOWN",
        "FIFI_FLASH_GGUF_TEST_F16_DELTA_OUT",
        "FIFI_FLASH_GGUF_TEST_DELTA_BENCH",
    ] {
        assert!(
            std::env::var_os(variable).is_none(),
            "{variable} changes the pinned baseline"
        );
    }
    let output = std::env::var_os("FIFI_FLASH_GGUF_STATEFUL_GPU_OUT").expect("new trace path");
    assert!(!Path::new(&output).exists(), "trace output must be new");
    let path = std::env::var_os("FIFI_FLASH_GGUF").expect("pinned first shard");
    assert_pinned_file(
        &path,
        "9d5644b371ffac921427c8021cc7242dab8e3ba37d7b48d1e79fe3770fbf9915",
    );
    let model = GgufModel::open(Path::new(&path)).unwrap().unwrap();
    let shape = model.config().model_shape().unwrap();
    assert!(model.config().layers() >= END && !model.config().kinds()[0]);
    let width = shape.hyper.hidden() * shape.hyper.branches();
    let host = pinned_host(width, model.config().layers());
    let ngram = model.ngram().unwrap();
    let mut feed = GgufTokenFeeds::new(&model, &ngram, PROMPT.len()).unwrap();
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let trunk = GgufTrunk::upload(&model, &ctx, &stream, 0..END).unwrap();
    let bound = GgufStep::new(&ctx, &stream, &trunk, PROMPT.len()).unwrap();
    let step = bound.step();
    let mut owner = GgufMoeOwner::new(&bound, shape.moe.top_k() + 1).unwrap();
    let mut trace = trace::StatefulTrace::new(END, width, shape.moe.top_k(), shape.moe.experts());
    for (position, token) in PROMPT.into_iter().enumerate() {
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
                        .flat_map(|x| x.to_ne_bytes())
                        .collect::<Vec<_>>(),
                )
                .unwrap()
        };
        write(&start.stream);
        for layer in 0..END {
            if layer == shape.ple_layer {
                let mut residual = step.read_f32(state, width).unwrap();
                let update = feed.ple_update(token, &residual).unwrap();
                assert_eq!(residual.len(), update.len());
                for (value, delta) in residual.iter_mut().zip(update) {
                    *value += delta;
                }
                write(&residual);
            }
            let mut snapshot = None;
            {
                let mut tap = trace::RouteTap::new(owner.bind(&bound).unwrap(), &mut snapshot);
                step.run_layer(layer, position, 1, &mut tap).unwrap();
            }
            trace.record(step, position, token, layer, snapshot.expect("route tap"));
            assert_eq!(step.health().unwrap(), (0, 0));
            if layer == 0 && selected(mask, position) {
                replace_stream(step, &host[position][0], position);
            }
        }
        step.finish_token().unwrap();
    }
    trace.finish(PROMPT.len());
    eprintln!("layer-0 intervention mask={mask} completed; diagnostic only, not logits parity");
}
