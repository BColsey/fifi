//! Token-forcing qualification of the Flint CUDA executor on the real EXL3 pack.
//!
//! One opt-in test runs the gate of `family::token_forcing` on the device:
//!
//! - the baseline: an executor opened with `TF_BASELINE_CHUNK` prefill rows generates greedily
//!   through its eager path (prompt prefill, then `qual_logits` steps, host argmax);
//! - the reference: an executor opened with `TF_REFERENCE_CHUNK` rows generates the same way.
//!   That executor is also the scorer, so its own rows judge every output;
//! - the candidate (`TF_CANDIDATE`): `serve`, the same executor's serving loop (`decode` with
//!   the device sampler); `fault`, the reference path with SpecPilot's injected logit bias after
//!   2,048 tokens, a self-test that must be rejected; or a generations file that another build
//!   wrote with `TF_WRITE_GENERATIONS`, for an A/B between builds on one pack.
//!
//! Every execution is opened with one EOS id that the head is not expected to choose (the last
//! tokenizer id), so each request runs to `TF_TOKENS` tokens: the gate tests numerics over long
//! outputs, and an early EOS would only shorten that evidence. The prompts are cut from the
//! committed corpus snapshot of the teacher-forced gate (`tf_gate_tests`), so no download is
//! needed. Audits run with the scorer only: a second executor does not fit beside it on a 24 GB
//! card, so the baseline-model audit is absent from device reports. Each audit is one prefill of
//! the whole context up to its event; a low `TF_AUDIT_CAP` cannot help a faulty candidate pass,
//! because an unaudited candidate event counts as unexplained.
//!
//! | Variable | Meaning | Default |
//! | --- | --- | --- |
//! | `FLINT_QUAL_MODEL` | the EXL3 pack root | required |
//! | `FLINT_QUAL_TOKENIZER` | the tokenizer directory | the pack root |
//! | `TF_OUT_DIR` | a new directory for the scored files and the report | required |
//! | `TF_KV` | the KV profile of every execution: `f16`, `q8` or `q8k` | `f16` |
//! | `TF_PROMPTS` | requests | 4 |
//! | `TF_PROMPT_TOKENS` | prompt length in tokens | 1024 |
//! | `TF_TOKENS` | generated tokens per request | 4096 |
//! | `TF_REFERENCE_CHUNK` | prefill rows of the reference and the scorer, 1 to 8 | 8 |
//! | `TF_BASELINE_CHUNK` | prefill rows of the baseline, 1 to 8 | 4 |
//! | `TF_CANDIDATE` | `serve`, `fault`, or the path of a generations file | `serve` |
//! | `TF_AUDIT_CAP` | the most events audited per execution | 200, SpecPilot's count |
//! | `TF_WRITE_GENERATIONS` | write this build's serving generations to this new file and stop | unset |
//!
//! ```text
//! FLINT_QUAL_MODEL=<PACK> TF_OUT_DIR=<NEW_DIR> \
//! cargo test --release --features native-cuda --lib \
//!   family::flint::cuda_executor::token_forcing_tests::token_forcing_qualification \
//!   -- --ignored --exact --nocapture
//! ```
//!
//! The run writes `reference.json`, `baseline.json`, `candidate.json` (scored outputs) and
//! `report.json`; `family::token_forcing::run::entry::token_forcing_verdict` recomputes the
//! verdict from the three scored files on any host.

use super::super::{factory, park_identity::pack_fingerprint};
use super::*;
use crate::family::token_forcing::{
    gate::{ExecutionIdentity, GateConfig, Generation, Verdict, ids_sha256},
    run::{Identities, Outputs, RunSettings, qualify, write_new_json},
    score::{ForcedModel, LogitBiasFault, greedy_generate},
};
use crate::tokenizer::{IncrementalDecoder, TokenizerBundle};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::PathBuf;

/// The executor's eager path as the gate's reference model.
struct Forced<'a> {
    engine: &'a mut FlintExecutor,
    session: &'a mut FlintSession,
}

impl ForcedModel for Forced<'_> {
    fn reset(&mut self) -> Result<()> {
        self.engine.reset(self.session)?;
        self.engine.boundary(self.session);
        Ok(())
    }

    fn prefill_logits(&mut self, context: &[u32]) -> Result<Vec<f32>> {
        let &last = context
            .last()
            .ok_or_else(|| EngError::BadRequest("an empty forced context".into()))?;
        // `prefill` feeds every token but the last, which stays pending; the logits row
        // consumes it.
        self.engine.prefill(self.session, context)?;
        self.engine.qual_logits(self.session, last)
    }

    fn step_logits(&mut self, token: u32) -> Result<Vec<f32>> {
        self.engine.qual_logits(self.session, token)
    }
}

#[derive(Clone, Debug)]
struct Plan {
    root: PathBuf,
    tokenizer: PathBuf,
    out: PathBuf,
    kv: String,
    prompts: usize,
    prompt_tokens: usize,
    tokens: usize,
    reference_chunk: u32,
    baseline_chunk: u32,
    candidate: String,
    audit_cap: usize,
    write_generations: Option<PathBuf>,
}

impl Plan {
    fn from_env() -> Self {
        let root: PathBuf = std::env::var_os("FLINT_QUAL_MODEL")
            .expect("FLINT_QUAL_MODEL must name the EXL3 pack root")
            .into();
        let number = |name: &str, default: usize| {
            std::env::var(name).map_or(default, |text| {
                text.parse()
                    .unwrap_or_else(|error| panic!("{name}={text}: {error}"))
            })
        };
        let chunk = |name: &str, default: usize| {
            let rows = number(name, default);
            assert!((1..=8).contains(&rows), "{name} must be 1 to 8, not {rows}");
            rows as u32
        };
        Self {
            tokenizer: std::env::var_os("FLINT_QUAL_TOKENIZER")
                .map_or_else(|| root.clone(), Into::into),
            out: std::env::var_os("TF_OUT_DIR")
                .expect("TF_OUT_DIR must name a new directory")
                .into(),
            kv: std::env::var("TF_KV").unwrap_or_else(|_| "f16".to_owned()),
            prompts: number("TF_PROMPTS", 4),
            prompt_tokens: number("TF_PROMPT_TOKENS", 1024),
            tokens: number("TF_TOKENS", 4096),
            reference_chunk: chunk("TF_REFERENCE_CHUNK", 8),
            baseline_chunk: chunk("TF_BASELINE_CHUNK", 4),
            candidate: std::env::var("TF_CANDIDATE").unwrap_or_else(|_| "serve".to_owned()),
            audit_cap: number("TF_AUDIT_CAP", 200),
            write_generations: std::env::var_os("TF_WRITE_GENERATIONS").map(Into::into),
            root,
        }
    }

    fn context_limit(&self) -> u32 {
        let rows = (self.prompt_tokens + self.tokens + 64).next_multiple_of(1024);
        u32::try_from(rows).expect("the context limit fits u32")
    }

    fn open(&self, chunk: u32, bundle: &TokenizerBundle) -> (FlintExecutor, FlintSession) {
        let options = FlintOptions {
            context_limit: self.context_limit(),
            kv_profile: self.kv.clone(),
            prefill_chunk_size: chunk,
            cpu_moe_offload_layers: 0,
            cpu_moe_split_experts: 0,
            cpu_moe_threads: 0,
        };
        let vocab = bundle.vocab_size() as usize;
        FlintExecutor::open(
            &self.root,
            &options,
            vec![never_chosen_eos(vocab)],
            vocab,
            factory::pinned_template_sha(),
        )
        .expect("the qualification executor (loud by design)")
    }

    fn identity(&self, label: &str, chunk: u32, path: &str) -> ExecutionIdentity {
        let fingerprint = pack_fingerprint(&self.root).expect("the pack fingerprint");
        let env = |name: &str| std::env::var(name).unwrap_or_default();
        let profile = BTreeMap::from([
            ("kv_profile".to_owned(), self.kv.clone()),
            ("prefill_chunk_size".to_owned(), chunk.to_string()),
            ("context_limit".to_owned(), self.context_limit().to_string()),
            ("generation_path".to_owned(), path.to_owned()),
            (
                "numeric_env".to_owned(),
                format!("{:?}", frozen_prefix_env()),
            ),
            (
                "FIFI_EXL3_LAYER_MAJOR".to_owned(),
                env("FIFI_EXL3_LAYER_MAJOR"),
            ),
            (SINGLE_GATHER_VAR.to_owned(), env(SINGLE_GATHER_VAR)),
            ("executable_sha256".to_owned(), executable_sha256()),
        ]);
        ExecutionIdentity {
            label: label.to_owned(),
            family: "flint-cuda".to_owned(),
            weights: hex(&fingerprint),
            weight_format: "exl3".to_owned(),
            profile,
        }
    }
}

/// The last tokenizer id: the head is not expected to choose it, so no request ends early.
fn never_chosen_eos(vocab: usize) -> u32 {
    u32::try_from(vocab - 1).expect("the vocabulary fits u32")
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn executable_sha256() -> String {
    static SHA: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    SHA.get_or_init(|| {
        let exe = std::env::current_exe().expect("the test executable path");
        let bytes = std::fs::read(&exe).expect("the test executable");
        format!("{:x}", Sha256::digest(bytes))
    })
    .clone()
}

/// Requests cut from the committed corpus, `prompt_tokens` each, one generated span apart.
fn prompts(plan: &Plan, bundle: &TokenizerBundle) -> Vec<(String, Vec<u32>)> {
    let ids = super::tf_gate_tests::corpus_ids(bundle.tokenizer());
    let stride = plan.prompt_tokens + plan.tokens;
    (0..plan.prompts)
        .map(|index| {
            let start = index * stride;
            let prompt = ids
                .get(start..start + plan.prompt_tokens)
                .unwrap_or_else(|| panic!("the corpus has no prompt {index} at id {start}"));
            (format!("corpus-{start}"), prompt.to_vec())
        })
        .collect()
}

fn generate_with(
    model: &mut dyn ForcedModel,
    prompts: &[(String, Vec<u32>)],
    tokens: usize,
    vocab: usize,
) -> Vec<Generation> {
    prompts
        .iter()
        .map(|(name, prompt)| Generation {
            name: name.clone(),
            prompt: prompt.clone(),
            generated: greedy_generate(model, prompt, tokens, vocab).expect("eager generation"),
        })
        .collect()
}

/// The serving loop: one prefill, then `decode` with the job's greedy sampler on the device.
fn serve(
    (engine, session): (&mut FlintExecutor, &mut FlintSession),
    bundle: &TokenizerBundle,
    prompts: &[(String, Vec<u32>)],
    tokens: usize,
) -> Vec<Generation> {
    let limit = u32::try_from(tokens).expect("the output limit fits u32");
    let mut generations = Vec::with_capacity(prompts.len());
    for (name, prompt) in prompts {
        engine.check_prompt(prompt.len()).expect("the prompt fits");
        engine.reset(session).expect("reset");
        engine.boundary(session);
        engine.prefill(session, prompt).expect("serving prefill");
        let mut state = GenerationState::new(IncrementalDecoder::new(bundle.tokenizer()))
            .with_max_tokens(limit);
        let mut generated = Vec::with_capacity(tokens);
        while generated.len() < tokens {
            let batch = engine.decode(session, &mut state).expect("serving decode");
            generated.extend_from_slice(&batch.tokens);
            engine.after_emit().expect("after emit");
            if batch.finish.is_some() {
                break;
            }
        }
        engine.boundary(session);
        generations.push(Generation {
            name: name.clone(),
            prompt: prompt.clone(),
            generated,
        });
    }
    generations
}

/// A build's serving outputs, for an A/B between two builds on one pack.
#[derive(Serialize, Deserialize)]
struct GenerationsFile {
    system: ExecutionIdentity,
    requests: Vec<GenerationRecord>,
}

#[derive(Serialize, Deserialize)]
struct GenerationRecord {
    name: String,
    prompt_sha256: String,
    generated: Vec<u32>,
}

fn read_generations(
    path: &std::path::Path,
    prompts: &[(String, Vec<u32>)],
) -> (ExecutionIdentity, Vec<Generation>) {
    let text = std::fs::read_to_string(path).expect("the candidate generations file");
    let file: GenerationsFile = serde_json::from_str(&text).expect("generations JSON");
    assert_eq!(
        file.requests.len(),
        prompts.len(),
        "the generations cover other prompts"
    );
    let generations = file
        .requests
        .into_iter()
        .zip(prompts)
        .map(|(record, (name, prompt))| {
            assert!(
                record.name == *name && record.prompt_sha256 == ids_sha256(prompt),
                "generation {} was made from another prompt",
                record.name
            );
            Generation {
                name: record.name,
                prompt: prompt.clone(),
                generated: record.generated,
            }
        })
        .collect();
    (file.system, generations)
}

#[test]
#[ignore = "needs the FLINT_QUAL_MODEL EXL3 pack and an exclusive GPU lease"]
fn token_forcing_qualification() {
    let plan = Plan::from_env();
    let bundle = factory::load_tokenizer(&plan.tokenizer).expect("the qualification tokenizer");
    let vocab = bundle.vocab_size() as usize;
    let prompts = prompts(&plan, &bundle);
    if let Some(path) = &plan.write_generations {
        let (mut engine, mut session) = plan.open(plan.reference_chunk, &bundle);
        let generations = serve((&mut engine, &mut session), &bundle, &prompts, plan.tokens);
        let requests = generations
            .into_iter()
            .map(|generation| GenerationRecord {
                prompt_sha256: ids_sha256(&generation.prompt),
                name: generation.name,
                generated: generation.generated,
            })
            .collect();
        let system = plan.identity("candidate-serve", plan.reference_chunk, "serving-decode");
        write_new_json(path, &GenerationsFile { system, requests }).expect("generations file");
        println!("TOKEN_FORCING generations={}", path.display());
        return;
    }
    std::fs::create_dir(&plan.out).expect("TF_OUT_DIR must not exist yet");
    let baseline = {
        let (mut engine, mut session) = plan.open(plan.baseline_chunk, &bundle);
        let mut model = Forced {
            engine: &mut engine,
            session: &mut session,
        };
        generate_with(&mut model, &prompts, plan.tokens, vocab)
    };
    let (mut engine, mut session) = plan.open(plan.reference_chunk, &bundle);
    let reference = {
        let mut model = Forced {
            engine: &mut engine,
            session: &mut session,
        };
        generate_with(&mut model, &prompts, plan.tokens, vocab)
    };
    let (candidate_identity, candidate) = match plan.candidate.as_str() {
        "serve" => (
            plan.identity("candidate-serve", plan.reference_chunk, "serving-decode"),
            serve((&mut engine, &mut session), &bundle, &prompts, plan.tokens),
        ),
        "fault" => {
            let mut model = LogitBiasFault::specpilot(Forced {
                engine: &mut engine,
                session: &mut session,
            });
            (
                plan.identity(
                    "candidate-fault",
                    plan.reference_chunk,
                    "eager-greedy-bias-fault",
                ),
                generate_with(&mut model, &prompts, plan.tokens, vocab),
            )
        }
        path => read_generations(std::path::Path::new(path), &prompts),
    };
    let identities = Identities {
        scorer: plan.identity("scorer", plan.reference_chunk, "eager-teacher-forced"),
        reference: plan.identity("reference", plan.reference_chunk, "eager-greedy"),
        baseline: plan.identity("baseline", plan.baseline_chunk, "eager-greedy"),
        candidate: candidate_identity,
    };
    let mut scorer = Forced {
        engine: &mut engine,
        session: &mut session,
    };
    let outputs = Outputs {
        reference: &reference,
        baseline: &baseline,
        candidate: &candidate,
    };
    let settings = RunSettings {
        config: GateConfig::QUALIFICATION,
        vocab,
        audit_cap: plan.audit_cap,
    };
    let run = qualify((&mut scorer, None), &identities, outputs, settings).expect("the gate runs");
    for (name, scored) in [
        ("reference.json", &run.reference),
        ("baseline.json", &run.baseline),
        ("candidate.json", &run.candidate),
    ] {
        write_new_json(&plan.out.join(name), scored).expect("a scored file");
    }
    let report_path = plan.out.join("report.json");
    write_new_json(&report_path, &run.report).expect("the report");
    println!(
        "TOKEN_FORCING candidate={} verdict={} report={}",
        plan.candidate,
        serde_json::to_string(&run.report.verdict).expect("a verdict serializes"),
        report_path.display()
    );
    if plan.candidate == "fault" {
        assert!(
            matches!(run.report.verdict, Verdict::Reject { .. }),
            "the injected fault was not rejected; see {}",
            report_path.display()
        );
    } else {
        assert_eq!(
            run.report.verdict,
            Verdict::Pass,
            "see {}",
            report_path.display()
        );
    }
}
