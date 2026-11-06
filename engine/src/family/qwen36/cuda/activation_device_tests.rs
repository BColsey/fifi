//! Explicit entry for a separately authorized real-weight smoke run. NOT an M2 qualification.
//! This harness checks the tokenizer digest; model/source identities remain declared inputs.
use super::{
    common_tests::{dims, inventory},
    dims::{KvMode, StepOptions},
    exl3_load::load_exl3_dir,
    resources::StepResources,
    rules::QwenRules,
    step::QwenResidentStep,
};
use crate::{
    family::{
        executor::{ExecutorOps, GenerationState},
        project::cuda::{CudaProjector, Reserve, WeightLayout},
        qwen36::activation::{CaptureRequest, Fingerprints, MAX_PROMPT_TOKENS},
        session::DecodeSession,
    },
    tokenizer::{IncrementalDecoder, Tokenizer},
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    io::{Read, Write},
    path::Path,
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    model_sha256: String,
    tokenizer_sha256: String,
    source_sha256: String,
    execution_sha256: String,
    prompt: Vec<u32>,
}

fn bounded_read(path: &Path, cap: usize) -> Vec<u8> {
    let mut bytes = Vec::new();
    std::fs::File::open(path)
        .expect("input file")
        .take((cap + 1) as u64)
        .read_to_end(&mut bytes)
        .expect("read input");
    assert!(bytes.len() <= cap, "input exceeds byte cap");
    bytes
}

#[test]
#[ignore = "explicit GPU authorization, exclusive window and frozen input manifest required"]
fn research_capture_final_prompt_block19() {
    assert_eq!(
        std::env::var("FIFI_ACTIVATION_GPU_AUTH").as_deref(),
        Ok("1"),
        "separate GPU authorization is required; never run as a host-only check"
    );
    let input_path =
        std::env::var_os("FIFI_ACTIVATION_MANIFEST").expect("frozen manifest required");
    let input: Input =
        serde_json::from_slice(&bounded_read(Path::new(&input_path), 2 * 1024 * 1024))
            .expect("valid input manifest");
    assert!(!input.prompt.is_empty() && input.prompt.len() <= MAX_PROMPT_TOKENS);
    let ids = Fingerprints::new(
        input.model_sha256,
        input.tokenizer_sha256.clone(),
        input.source_sha256,
    )
    .expect("frozen identities");
    let request = CaptureRequest::new(
        ids,
        &input.prompt,
        dims().shape.hidden,
        input.execution_sha256,
    )
    .expect("capture request");
    let output =
        std::env::var_os("FIFI_ACTIVATION_ARTIFACT").expect("explicit artifact path required");
    let output = Path::new(&output);
    assert!(!output.exists(), "fresh artifact filename required");
    assert!(
        output.parent().is_some_and(Path::is_dir),
        "artifact parent must exist"
    );
    let dir = std::env::var_os("FIFI_EXL3_CHECKPOINT").expect("explicit local checkpoint required");
    let tok_bytes = bounded_read(&Path::new(&dir).join("tokenizer.json"), 64 * 1024 * 1024);
    assert_eq!(
        format!("{:x}", Sha256::digest(&tok_bytes)),
        input.tokenizer_sha256
    );
    let tokenizer = Tokenizer::from_bytes(&tok_bytes).expect("tokenizer");
    let projector = CudaProjector::new().expect("CUDA unavailable");
    let weights =
        load_exl3_dir(&projector, Path::new(&dir), Reserve::default()).expect("EXL3 unavailable");
    let options = StepOptions {
        kv_mode: KvMode::Int8G32,
        context_tokens: input.prompt.len() + 2,
        delta_slots: 1,
        forked_experts: false,
        tiered: None,
        layout: WeightLayout::Exl3,
    };
    let resources =
        StepResources::new(projector.context(), dims(), &inventory(), &options).expect("resources");
    let rules = QwenRules::new(
        vec![248_046, 248_044],
        tokenizer.vocab_size() as usize,
        248_320,
    )
    .expect("rules");
    let mut step =
        QwenResidentStep::new(projector.context(), &resources, &weights, &options, rules)
            .expect("resident step");
    let mut session = DecodeSession::default();
    ExecutorOps::prefill(&mut step, &mut session, &input.prompt).expect("prefill");
    let mut generation = GenerationState::new(IncrementalDecoder::new(&tokenizer));
    let (_, record) = step
        .decode_with_activation(&mut session, &mut generation, request)
        .expect("capture");
    assert_eq!(step.tokens(), input.prompt.len(), "no second forward");
    let bytes = record.to_json().expect("bounded artifact");
    let mut file = tempfile::NamedTempFile::new_in(output.parent().expect("artifact parent"))
        .expect("local output");
    file.write_all(&bytes).expect("write artifact");
    file.persist_noclobber(output)
        .expect("fresh artifact filename required");
}
