//! The device against the host on the small model: the same tokens through the CUDA driver
//! (trunk upload, native projections, DeltaNet scan in the tiled order, native MoE block, host
//! PLE feed) and through the host forward over the same file. The weights are one set of bytes;
//! the device quantizes its activations to q8_1 and widens some tensors to half precision, so the
//! logits agree to a tolerance and the arg-max agrees exactly.
//!
//! Each test needs the card and holds a CUDA adapter lease for its process, and a process holds
//! at most two: run them by name, one per process, with `--ignored`.

use super::{
    GgufModel, host,
    smallmodel::{Small, write_small},
};
use crate::family::{
    flash_next::cuda::{
        driver::TokenDriver,
        gguf_device::{GgufDevice, gguf_driver},
        pack_feeds::PackFeeds,
    },
    project::cuda::CudaProjector,
    session::TokenStepper,
};

/// The driver over the small model, owning its model, card handle and weights.
fn driver_of(small: &Small, positions: usize) -> TokenDriver<GgufDevice, PackFeeds> {
    let model = GgufModel::open(&small.first)
        .expect("open")
        .expect("a qwen4exp model");
    let projector = CudaProjector::new().expect("the card");
    gguf_driver(model, projector, 8 << 20, positions, 2).expect("the driver")
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .expect("nonempty")
}

/// How far the device logits sit from the host's over a token sequence.
struct Agreement {
    /// The largest absolute difference at any position.
    worst: f32,
    /// The largest ratio of a position's difference to its bound.
    worst_ratio: f32,
    /// Positions whose top-1 matched, and positions whose host margin was too small to ask.
    matched: usize,
    undecided: usize,
    /// Positions where a selection was close enough to flip: the run stops at the first.
    flippable: usize,
    /// Positions whose difference was held to the bound.
    compared: usize,
}

/// The activation rounding step of the device, as a fraction of a block's largest value: a q8_1
/// block stores 8-bit integers at scale `amax / 127`, so a value is off by at most half a step.
const Q8_HALF_STEP: f32 = 0.5 / 127.0;

/// The bound on a logit's difference at a position whose host logits reach `scale`. Every
/// quantized projection (about twelve a layer and the head) adds an independent relative error of
/// at most half a q8_1 step; independent errors add in quadrature, and the logit row is a
/// linear read-out of values of the size of `scale`.
fn bound(layers: usize, scale: f32) -> f32 {
    Q8_HALF_STEP * ((12 * layers + 1) as f32).sqrt() * scale
}

/// Runs `tokens` through the device and the host forward over the small model of `layers`
/// layers and compares every position: each difference stays under [`bound`], and where the
/// host's top-1 leads the second by more than twice the bound the device picks the same id.
fn agreement(small: &Small, layers: usize, tokens: &[u32]) -> Agreement {
    let model: &'static GgufModel = Box::leak(Box::new(
        GgufModel::open(&small.first)
            .expect("open")
            .expect("a qwen4exp model"),
    ));
    let mut driver = driver_of(small, tokens.len() + 8);
    let reference = host::HostModel::load(model, layers).expect("host layers");
    crate::family::qwen::flash_next::qsa::margins::take();
    let want = host::run(model, &reference, tokens, 2).expect("host forward");
    // A selection whose last kept block leads the first dropped one by less than twice the
    // bound's relative error can flip under the device's activation rounding; a flip changes
    // the attended positions, so the position is not a test of the arithmetic.
    let flip_gap = 2.0 * f64::from(bound(layers, 1.0));
    let mut sensitive = std::collections::BTreeSet::new();
    for (pos, gap) in crate::family::qwen::flash_next::qsa::margins::take() {
        if gap < flip_gap {
            sensitive.insert(pos as usize);
        }
    }
    let mut out = Agreement {
        worst: 0.0,
        worst_ratio: 0.0,
        matched: 0,
        undecided: 0,
        flippable: 0,
        compared: 0,
    };
    for (at, &token) in tokens.iter().enumerate() {
        let got = driver.logits(token).expect("device logits");
        assert_eq!(got.len(), want.logits[at].len());
        assert!(got.iter().all(|v| v.is_finite()), "position {at}");
        let diff = got
            .iter()
            .zip(&want.logits[at])
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let scale = host_row.iter().fold(0f32, |m, v| m.max(v.abs()));
        let limit = bound(layers, scale);
        if sensitive.contains(&at) {
            eprintln!(
                "position {at}: a selection within the flip gap, difference {diff} not asserted"
            );
            out.flippable += 1;
            // Past a flip the recurrent state differs, so no later position tests the arithmetic.
            break;
        }
        out.compared += 1;
        assert!(
            diff <= limit,
            "position {at}: difference {diff} over the bound {limit} (scale {scale})"
        );
        assert_eq!(argmax(&got), argmax(&want.logits[at]), "position {at}");
    }
    eprintln!(
        "{} positions, {layers} layers: worst difference {}, worst ratio to the bound {:.3}, top-1 matched {}, undecided {}, compared {}, flippable {}",
        tokens.len(),
        out.worst,
        out.worst_ratio,
        out.matched,
        out.undecided,
        out.compared,
        out.flippable
    );
    out
}

fn sequence(len: usize) -> Vec<u32> {
    (0..len as u32).map(|i| (i * 37 + 11) % 512).collect()
}

#[test]
#[ignore = "needs a GPU (about 2 GiB)"]
fn the_device_logits_follow_the_host_forward_on_the_small_model() {
    let small = write_small(7);
    let got = agreement(&small, 4, &[5, 9, 200, 17, 33, 481]);
    assert!(got.matched >= 2, "{} of 6 positions decided", got.matched);
}

/// A prompt past the indexer's budget (4 positions, the block ratio of the model) over two QSA
/// layers: from the eighth token on the selection drops blocks. The run is held to the bound up
/// to the first selection that the device's rounding may flip.
#[test]
#[ignore = "needs a GPU (about 2 GiB)"]
fn the_device_follows_the_host_when_the_indexer_drops_positions() {
    let small = write_small_with(7, 8, 4);
    let got = agreement(&small, 8, &sequence(48));
    // Dropping starts at position 7: the positions before the first flippable selection hold
    // at least ten of them.
    assert!(got.compared >= 17, "{} positions compared", got.compared);
}

/// More than a thousand positions deep, over two QSA layers whose indexer keeps every block
/// (a budget past the depth), so no selection can flip and every position is held to the bound.
#[test]
#[ignore = "needs a GPU (about 2 GiB)"]
fn the_device_follows_the_host_a_thousand_positions_deep() {
    let small = write_small_with(7, 8, 2048);
    let got = agreement(&small, 8, &sequence(1030));
    assert_eq!(got.compared, 1030, "every position is compared");
    assert!(
        got.matched >= 400,
        "{} of 1030 positions decided",
        got.matched
    );
}

fn time_run<S: TokenStepper>(driver: &mut S) -> f64 {
    driver.reset().expect("reset");
    let start = std::time::Instant::now();
    for at in 0..48u32 {
        let logits = driver.logits(5 + at * 7 % 400).expect("logits");
        assert!(logits.iter().all(|v| v.is_finite()));
    }
    start.elapsed().as_secs_f64() * 1e3 / 48.0
}

/// Decode time per token on the small model, which picks ten experts per layer: the median of
/// five runs of 48 tokens after one warm-up run. It prints the figure; the check is that every
/// run completes and the logits stay finite.
#[test]
#[ignore = "needs a GPU (about 2 GiB)"]
fn the_decode_time_per_token_on_the_small_model() {
    let small = write_small(7);
    let mut driver = driver_of(&small, 128);
    time_run(&mut driver);
    let mut times: Vec<f64> = (0..5).map(|_| time_run(&mut driver)).collect();
    times.sort_by(f64::total_cmp);
    eprintln!(
        "decode ms per token on the small model: median {:.3}, runs {times:.3?}",
        times[2]
    );
}

/// A tokenizer of eight single-letter tokens, the shape the family tests use.
const TOKENIZER_JSON: &str = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"post_processor":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":{"a":0,"b":1,"c":2,"d":3,"e":4,"f":5,"g":6,"h":7},"merges":[]}}"#;

/// A default start of the engine factory on a directory holding the small model and a
/// tokenizer: the plan, the upload and the executor are the served ones, and the greedy tokens
/// of a job equal the host forward's own greedy choice over the tokenizer's eight ids.
#[test]
#[ignore = "needs a GPU (about 2 GiB)"]
fn an_engine_started_on_the_small_gguf_serves_the_greedy_tokens_of_the_host_forward() {
    use crate::{
        family::{executor::GenerationState, flash_next::factory, sampler::Sampler},
        sched::{
            job::{Job, JobEvent},
            worker::EngineNote,
        },
        tokenizer::{IncrementalDecoder, Tokenizer},
    };
    const NEW_TOKENS: usize = 6;
    let small = write_small(7);
    let dir = small.dir.path();
    for (name, text) in [
        ("tokenizer.json", TOKENIZER_JSON),
        (
            "tokenizer_config.json",
            r#"{"eos_token":"h","chat_template":"{{ messages[0].content }}"}"#,
        ),
    ] {
        std::fs::write(dir.join(name), text).expect("tokenizer file");
    }
    let prompt = [1u32, 2, 3];

    let model: &'static GgufModel = Box::leak(Box::new(
        GgufModel::open(&small.first)
            .expect("open")
            .expect("a qwen4exp model"),
    ));
    let reference = host::HostModel::load(model, 4).expect("host layers");
    let mut tokens = prompt.to_vec();
    let mut want = Vec::new();
    for _ in 0..NEW_TOKENS {
        let run = host::run(model, &reference, &tokens, 2).expect("host forward");
        let last = run.logits.last().expect("logits");
        let next = argmax(&last[..8]) as u32;
        want.push(next);
        tokens.push(next);
    }

    let (tx, rx) = std::sync::mpsc::channel();
    let started = factory::start_engine(
        dir,
        dir,
        None,
        false,
        Some(Box::new(move |note| tx.send(note).expect("note"))),
    )
    .expect("the engine starts");
    if let EngineNote::Failed(why) = rx.recv().expect("a note") {
        panic!("the start failed: {why}");
    }
    let tokenizer = Tokenizer::from_bytes(TOKENIZER_JSON.as_bytes()).expect("tokenizer");
    let state =
        GenerationState::with_sampler(IncrementalDecoder::new(&tokenizer), Sampler::greedy())
            .with_max_tokens(NEW_TOKENS as u32);
    let (job, mut response) = Job::new(prompt.to_vec(), state).expect("job");
    started.engine.handle().try_submit(job).expect("submit");
    let mut got = Vec::new();
    while let Some(event) = response.events.blocking_recv() {
        match event {
            JobEvent::Batch(batch) => got.extend(batch.tokens),
            JobEvent::Error(error) => panic!("{error:?}"),
            JobEvent::Started | JobEvent::Finished(_) => {}
        }
    }
    eprintln!("engine tokens {got:?}, host greedy tokens {want:?}");
    assert!(started.engine.shutdown().failure.is_none());
    assert_eq!(got, want);
}

/// Stopping the engine returns the card memory it took. The reading before the start is taken
/// with a card handle already open, so the context's own bytes are in both readings.
#[test]
#[ignore = "needs a GPU (about 2 GiB)"]
fn stopping_the_engine_gives_back_its_card_memory() {
    use crate::{family::flash_next::factory, sched::worker::EngineNote};
    const SLACK: usize = 256 << 20;
    let small = write_small(7);
    let dir = small.dir.path();
    for (name, text) in [
        ("tokenizer.json", TOKENIZER_JSON),
        (
            "tokenizer_config.json",
            r#"{"eos_token":"h","chat_template":"{{ messages[0].content }}"}"#,
        ),
    ] {
        std::fs::write(dir.join(name), text).expect("tokenizer file");
    }
    let probe = CudaProjector::new().expect("the card");
    let before = probe.memory_info().expect("memory").free;
    let live_before = crate::kernels::cuda::live_device_bytes();
    let (tx, rx) = std::sync::mpsc::channel();
    let started = factory::start_engine(
        dir,
        dir,
        None,
        false,
        Some(Box::new(move |note| tx.send(note).expect("note"))),
    )
    .expect("the engine starts");
    if let EngineNote::Failed(why) = rx.recv().expect("a note") {
        panic!("the start failed: {why}");
    }
    let during = probe.memory_info().expect("memory").free;
    let live_during = crate::kernels::cuda::live_device_bytes();
    assert!(started.engine.shutdown().failure.is_none());
    let after = probe.memory_info().expect("memory").free;
    let live_after = crate::kernels::cuda::live_device_bytes();
    eprintln!(
        "live device bytes: before {live_before}, running {live_during}, stopped {live_after}"
    );
    assert!(
        live_during > live_before,
        "the running engine holds buffers"
    );
    assert_eq!(
        live_after, live_before,
        "the stopped engine freed every buffer"
    );
    eprintln!(
        "card free MiB: before {}, running {}, stopped {}",
        before >> 20,
        during >> 20,
        after >> 20
    );
    assert!(
        after + SLACK >= before,
        "the stopped engine kept {} MiB",
        (before - after) >> 20
    );
}

/// Where the first difference sits: the residual stream after every layer, device against host,
/// at the positions around the first one where the indexer drops blocks.
#[test]
#[ignore = "needs a GPU (about 2 GiB)"]
fn which_layer_first_differs_when_the_indexer_drops_positions() {
    let small = write_small_with(7, 8, 4);
    let model: &'static GgufModel = Box::leak(Box::new(
        GgufModel::open(&small.first)
            .expect("open")
            .expect("a qwen4exp model"),
    ));
    let tokens = sequence(27);
    let mut driver = driver_of(&small, tokens.len() + 8);
    let reference = host::HostModel::load(model, 8).expect("host layers");
    let want = host::run(model, &reference, &tokens, 2).expect("host forward");
    crate::family::qwen::flash_next::cuda::gguf_device::layer_trace::start();
    for (at, &token) in tokens.iter().enumerate() {
        driver.logits(token).expect("device logits");
        let streams = crate::family::qwen::flash_next::cuda::gguf_device::layer_trace::take();
        let mut line = format!("position {at}:");
        for (layer, got) in streams.iter().enumerate() {
            let host_stream = &want.layer_out[at][layer];
            let scale = host_stream.iter().fold(0f32, |m, v| m.max(v.abs()));
            let diff = got
                .iter()
                .zip(host_stream)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            line.push_str(&format!(" L{layer} {:.4}", diff / scale.max(1e-6)));
        }
        eprintln!("{line}");
    }
}
