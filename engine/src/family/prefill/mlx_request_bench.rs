//! Test-only measurement of isolated worker requests with the same timing boundaries as reqlog.

use std::time::{Duration, Instant};

use crate::{
    family::executor::GenerationState,
    kernels::MlxKernels,
    sched::{
        job::{Job, JobEvent},
        worker::{Engine, EngineNote, EngineState, ReadyNote},
    },
};

pub(crate) const CACHE_LIMIT: u64 = 16 << 20;
const COMPLETION: u32 = 8;
const WARMUPS: usize = 2;

#[derive(Clone, Copy)]
pub(crate) struct Workload {
    pub(crate) piece: u32,
    pub(crate) context: u32,
    prompts: [usize; 2],
}

impl Workload {
    pub(crate) fn from_env() -> Self {
        let workload = match std::env::var("FIFI_MLX_REQUEST_PIECE") {
            Ok(value) => Self {
                piece: value.parse().expect("positive benchmark worker piece"),
                context: 512,
                prompts: [65, 257],
            },
            Err(std::env::VarError::NotPresent) => Self {
                piece: 8,
                context: 128,
                prompts: [8, 33],
            },
            Err(error) => panic!("benchmark worker piece: {error}"),
        };
        assert!(workload.piece > 0);
        workload
    }
}

/// Start a fresh single-session worker, warm each prompt size, and emit one JSON record per job.
pub(crate) fn measure(
    family: &str,
    arm: &str,
    kv_dtype: &str,
    workload: Workload,
    start: impl FnOnce(ReadyNote) -> Engine,
    prompt: impl Fn(usize) -> Vec<u32>,
    generation: impl Fn(u32) -> GenerationState,
) {
    let _serial = crate::native_test_lock();
    let samples = std::env::var("FIFI_MLX_REQUEST_SAMPLES")
        .map(|value| value.parse::<usize>().expect("positive sample count"))
        .unwrap_or(8);
    assert!(samples > 0);
    let mut counter = MlxKernels::try_new().unwrap();
    let (tx, rx) = std::sync::mpsc::channel();
    let engine = start(Some(Box::new(move |note| {
        tx.send(note).expect("benchmark ready receiver")
    })));
    match rx
        .recv_timeout(Duration::from_secs(30))
        .expect("worker startup")
    {
        EngineNote::Ready(_) => {}
        EngineNote::Failed(why) => panic!("{why}"),
    }
    let handle = engine.handle();

    for prompt_tokens in workload.prompts {
        let ids = prompt(prompt_tokens);
        assert_eq!(ids.len(), prompt_tokens);
        for iteration in 0..(WARMUPS + samples) {
            let (job, mut response) = Job::new(ids.clone(), generation(COMPLETION)).unwrap();
            let active_before = counter.memory_stats().unwrap().active;
            counter.reset_peak_memory().unwrap();
            counter.qualification_reset_timing();
            counter.qualification_reset_flush_stats();
            let before = counter.qualification_sync_count();
            let admitted = Instant::now();
            handle.try_submit(job).unwrap();
            let mut started = None;
            let mut first = None;
            let mut last = None;
            let mut tokens = Vec::new();
            let mut usage = None;
            while let Some(event) = response.events.blocking_recv() {
                let now = Instant::now();
                match event {
                    JobEvent::Started => assert!(started.replace(now).is_none()),
                    JobEvent::Batch(batch) => {
                        // Both fixture workers publish one token per decode step.
                        assert_eq!(batch.tokens.len(), 1);
                        first.get_or_insert(now);
                        last = Some(now);
                        tokens.extend(batch.tokens);
                    }
                    JobEvent::Finished(done) => assert!(usage.replace(done).is_none()),
                    JobEvent::Error(error) => panic!("{family} {arm}: {error:?}"),
                }
            }

            let usage = usage.expect("request reached Finished");
            let first = first.expect("first token");
            let last = last.expect("last token");
            assert_eq!(usage.prompt_tokens as usize, prompt_tokens);
            assert_eq!(usage.completion_tokens, COMPLETION);
            assert_eq!(tokens.len(), COMPLETION as usize);
            assert_eq!(usage.reused_tokens, 0);
            assert_eq!(
                usage.prefill_pieces.count as usize,
                prompt_tokens.div_ceil(workload.piece as usize)
            );
            assert_eq!(
                usage.prefill_pieces.max_rows as usize,
                prompt_tokens.min(workload.piece as usize)
            );
            let syncs = counter.qualification_sync_count() - before;
            let peak = counter.memory_stats().unwrap().peak;
            assert!(syncs > 0 && peak > 0);
            if let Some(sample) = iteration.checked_sub(WARMUPS) {
                let timing = counter.qualification_timing();
                let flush = counter.qualification_flush_stats();
                println!(
                    "mlx_request {}",
                    serde_json::json!({
                        "family": family,
                        "arm": arm,
                        "kv_dtype": kv_dtype,
                        "worker_piece": workload.piece,
                        "sample": sample,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": COMPLETION,
                        "prefill_pieces": usage.prefill_pieces.count,
                        "ttft_ns": first.duration_since(admitted).as_nanos(),
                        "start_to_first_ns": first.duration_since(started.expect("job started")).as_nanos(),
                        "decode_ns": last.duration_since(first).as_nanos(),
                        "metal_syncs": syncs,
                        "import_calls": timing.import_calls,
                        "import_ns": timing.import_ns,
                        "readback_calls": timing.readback_calls,
                        "readback_ns": timing.readback_ns,
                        "evaluate_calls": timing.evaluate_calls,
                        "evaluate_ns": timing.evaluate_ns,
                        "retire_flushes": flush.flushes,
                        "retire_flush_sync_ns": flush.sync_ns,
                        "mlx_active_before_bytes": active_before,
                        "mlx_peak_active_bytes": peak,
                        "tokens": tokens,
                    })
                );
            }
        }
    }

    let stopped = engine.shutdown();
    assert_eq!(stopped.state, EngineState::Stopped);
    assert!(stopped.failure.is_none());
}
