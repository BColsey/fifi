//! Release measurement of #68's seeded exact-retry through the public API, with the
//! library built WITHOUT cfg(test) so the resend cross-check stays off in production
//! form (a unit-test build re-encodes on every hit to verify itself; that is the
//! diagnostic, not the production cost).
//!
//! One process per config:
//!   cargo test --release --locked --test retry_bench -- --nocapture --test-threads=1
//! with ENGINESRV_SEED_RESENDS=1 for the on-config, unset for off.

use inference_engine::tokenizer::PromptMessage;
use inference_engine::tokenizer::Role;
use inference_engine::tokenizer::Thinking;
use inference_engine::tokenizer::TokenizerBundle;
use inference_engine::tokenizer::TokenizerPin;
use inference_engine::tokenizer::ToolSpec;
use inference_engine::tokenizer::WarmTurnCache;

fn hex(text: &str) -> [u8; 32] {
    let mut out = [0u8; 32];
    for (slot, pair) in out.iter_mut().zip(text.as_bytes().as_chunks::<2>().0) {
        *slot = u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap();
    }
    out
}

fn pin() -> TokenizerPin {
    TokenizerPin {
        tokenizer_json: hex("87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4"),
        tokenizer_config_json: hex(
            "e98f1901ac6f0adff67b1d540bfa0c36ac1a0cf59eb72ed78146ef89aafa1182",
        ),
        chat_template_jinja: hex(
            "e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259",
        ),
        generation_config_json: hex(
            "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e",
        ),
        vocab_size: 248_077,
    }
}

fn msgs(turns: usize) -> Vec<PromptMessage> {
    let para = "The staging ring reads each layer's misses as one batched pass at pooled depth. \
                An admission probe sizes the plan from the budget the cgroup actually grants, and \
                a refused start states its byte counts instead of falling back silently. The \
                prefix store keeps one block checkpoint per numeric path, cut at a token boundary \
                just before the end of the conversation, so a client that rewrites only the tail \
                still reuses everything before the change. Quantized experts keep their scales \
                in thirty-two value groups and pay one magnitude reduction per group.";
    let mut msgs = vec![PromptMessage {
        role: Role::System,
        content: Some("You are a careful assistant.".to_string()),
        reasoning_content: None,
        name: None,
        tool_call_id: None,
        tool_calls: Vec::new(),
    }];
    for i in 0..turns {
        let role = if i % 2 == 0 {
            Role::User
        } else {
            Role::Assistant
        };
        msgs.push(PromptMessage {
            role,
            content: Some(format!("{para} {para} Turn {i} says {para}.")),
            reasoning_content: None,
            name: None,
            tool_call_id: None,
            tool_calls: Vec::new(),
        });
    }
    msgs
}

/// One command reproduces the release table (39K and 256K tokens, full encode
/// against the seeded retry, plus the appended turn and the seed-time digest cost):
///
///   cargo test --release --locked --test retry_bench -- --ignored --nocapture --test-threads=1
///
/// ENGINESRV_SEED_RESENDS unset runs the default (on); `0` runs the off-config.
/// The local model directory is only READ, never written.
#[test]
#[ignore = "release bench; needs the pinned tokenizer bundle on disk"]
fn retry_bench() {
    let Some(dir) = std::env::var_os("Q36_CUDA_MODEL_DIR")
        .map(std::path::PathBuf::from)
        .inspect(|p| {
            assert!(
                p.is_dir(),
                "Q36_CUDA_MODEL_DIR={} is not a directory",
                p.display()
            )
        })
    else {
        eprintln!(
            "retry_bench: Q36_CUDA_MODEL_DIR is unset; skipping (the pinned tokenizer bundle is not on this host)"
        );
        return;
    };
    let bundle = TokenizerBundle::load(&dir, &pin()).expect("bundle");
    let seed_on =
        std::env::var_os("ENGINESRV_SEED_RESENDS").as_deref() != Some(std::ffi::OsStr::new("0"));
    let tools: Vec<ToolSpec> = Vec::new();
    let thinking = Thinking::Default;
    for turns in [120, 781] {
        let conversation = msgs(turns);
        let appended = {
            let mut longer = conversation.clone();
            longer.push(PromptMessage {
                role: Role::User,
                content: Some("One more appended user turn.".to_string()),
                reasoning_content: None,
                name: None,
                tool_call_id: None,
                tool_calls: Vec::new(),
            });
            longer
        };
        let full = {
            let t0 = std::time::Instant::now();
            let ids = bundle
                .encode_prompt(&conversation, &tools, thinking)
                .expect("full encode");
            (t0.elapsed().as_secs_f64() * 1000.0, ids.len())
        };
        let mut cache = WarmTurnCache::new();
        let (ids0, _) = bundle
            .encode_prompt_warm(&mut cache, &conversation, &tools, thinking)
            .expect("seed turn");
        let mut hits = Vec::new();
        for _ in 0..5 {
            let t0 = std::time::Instant::now();
            let (ids, _) = bundle
                .encode_prompt_warm(&mut cache, &conversation, &tools, thinking)
                .expect("retry");
            assert_eq!(ids, ids0, "retry ids identical");
            hits.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        hits.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // The appended turn (the common case): the cache holds the shorter
        // conversation, the request extends it by one turn. Only the FIRST
        // call after seeding is an append; repeating the same request would
        // measure the retry path instead. Each round re-seeds a fresh cache
        // with the shorter conversation, then times one extended request.
        let append_warm = {
            let mut t = Vec::new();
            for _ in 0..5 {
                let mut fresh = WarmTurnCache::new();
                let _ = bundle
                    .encode_prompt_warm(&mut fresh, &conversation, &tools, thinking)
                    .expect("re-seed");
                let t0 = std::time::Instant::now();
                let (ids, _) = bundle
                    .encode_prompt_warm(&mut fresh, &appended, &tools, thinking)
                    .expect("append");
                t.push((t0.elapsed().as_secs_f64() * 1000.0, ids.len()));
            }
            t.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            t[2]
        };
        // Seed-time digest cost, measured through the public API: a FRESH cache's
        // first (miss) turn with the seed on, minus the same with the seed off. The
        // off-process child pays no digest; the delta is what a non-retry request
        // pays for arming the switch.
        let seed_miss_ms = {
            let mut fresh = WarmTurnCache::new();
            let t0 = std::time::Instant::now();
            let _ = bundle
                .encode_prompt_warm(&mut fresh, &conversation, &tools, thinking)
                .expect("seed turn");
            t0.elapsed().as_secs_f64() * 1000.0
        };
        eprintln!(
            "RETRY_BENCH turns={turns} tokens={} seed_resends={seed_on} full_encode_ms={:.3} retry_median_ms={:.3} reuses={} seed_miss_ms={:.3} slot_bytes={} append_tokens={} append_median_ms={:.3}",
            full.1,
            full.0,
            hits[2],
            cache.reuses(),
            seed_miss_ms,
            cache.accounted_bytes(),
            append_warm.1,
            append_warm.0,
        );
    }
}
