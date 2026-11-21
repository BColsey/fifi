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

#[test]
#[ignore = "needs a GPU (about 2 GiB)"]
fn the_device_logits_follow_the_host_forward_on_the_small_model() {
    let small = write_small(7);
    let model: &'static GgufModel = Box::leak(Box::new(
        GgufModel::open(&small.first)
            .expect("open")
            .expect("a qwen4exp model"),
    ));
    let mut driver = driver_of(&small, 64);

    let tokens = [5u32, 9, 200, 17, 33, 481];
    let reference = host::HostModel::load(model, 4).expect("host layers");
    let want = host::run(model, &reference, &tokens, 2).expect("host forward");
    let mut worst = 0f32;
    let mut scale = 0f32;
    for (at, &token) in tokens.iter().enumerate() {
        let got = driver.logits(token).expect("device logits");
        assert_eq!(got.len(), want.logits[at].len());
        assert!(got.iter().all(|v| v.is_finite()), "position {at}");
        let diff = got
            .iter()
            .zip(&want.logits[at])
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        worst = worst.max(diff);
        scale = scale.max(want.logits[at].iter().fold(0f32, |m, v| m.max(v.abs())));
        eprintln!(
            "position {at}: device top-1 {}, host top-1 {}, max abs difference {diff}",
            argmax(&got),
            argmax(&want.logits[at])
        );
        assert_eq!(argmax(&got), argmax(&want.logits[at]), "position {at}");
    }
    eprintln!("worst max abs difference {worst} on logits up to {scale}");
    assert!(worst < 0.05 * scale.max(1.0), "{worst} against {scale}");
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
