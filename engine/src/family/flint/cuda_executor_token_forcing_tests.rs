//! Token-forcing qualification of the Flint CUDA executor on the real EXL3 pack.
//!
//! One opt-in test runs the gate of `family::token_forcing` on the device:
//!
//! - the baseline: an executor opened with `TF_BASELINE_CHUNK` prefill rows generates greedily
//!   through its eager path (prompt prefill, then `qual_logits` steps, host argmax);
//! - the reference: an executor opened with `TF_REFERENCE_CHUNK` rows generates the same way.
//!   That executor is also the scorer, so its own rows judge every output;
//! - the candidates (`TF_CANDIDATE`, comma-separated, judged in that order against the one
//!   scored reference and baseline): `serve`, the same executor's serving loop (`decode` with
//!   the device sampler); `fault`, the reference path with SpecPilot's injected logit bias after
//!   2,048 tokens, a self-test that must be rejected; or a generations file that another build
//!   wrote with `TF_WRITE_GENERATIONS`, for an A/B between builds on one pack.
//!
//! Flint's prefill is batch-invariant: chunk 1 and chunk 8 agree bitwise
//! (`context_2048_same_pack_bitwise_golden`). The baseline is then expected never to depart
//! from the reference, and the calibration sets the smallest delta, 1/8 nat, the strictest
//! gate.
//!
//! Every execution is opened with one EOS id that the head is not expected to choose (the last
//! tokenizer id), so each request runs to `TF_TOKENS` tokens: the gate tests numerics over long
//! outputs, and an early EOS would only shorten that evidence.
//!
//! The prompts are the committed tasks of `testdata/token-forcing-tasks.json`, each one user
//! turn through the pack's chat template, asking for a long coding or derivation answer.
//! Raw-text continuations of the teacher-forced gate's corpus were tried first and cannot
//! qualify anything: by 2,048 tokens every output had become a repetition loop (2 to 35
//! distinct ids, median top-two gap 8.6 nats), the injected fault changed no token, and the
//! self-test passed a faulty candidate. `TF_PROBE=1` measures that risk before a
//! qualification: it generates the reference output of each selected task and writes
//! `probe.json`. From the fault's start up to the first id that would end the answer, it
//! counts the distinct ids, the positions whose top two logits lie closer than the fault's
//! bias, the positions where the bias changes the argmax (`fault_flips`), and those of them
//! whose margin exceeds twice the largest delta the gate accepts (`fault_events`): events
//! under any calibration, which the unaudited fault cannot explain. The counts read the
//! reference's own history; the fault leaves it at its first flip, so they estimate its reach
//! rather than predict its mismatches. They are zero when `TF_TOKENS` is at most 2,048, and
//! text after an end id does not count, because the run continues past it (see above).
//!
//! Audits run with the scorer only: a second executor does not fit beside it on a 24 GB
//! card, so the baseline-model audit is absent from device reports. Each audit is one prefill of
//! the whole context up to its event, so the `fault` self-test is not audited: its events
//! count as unexplained, and auditing hundreds of them would take hours. A low `TF_AUDIT_CAP`
//! cannot help a faulty candidate pass either, for the same reason.
//!
//! | Variable | Meaning | Default |
//! | --- | --- | --- |
//! | `FLINT_QUAL_MODEL` | the EXL3 pack root | required |
//! | `FLINT_QUAL_TOKENIZER` | the tokenizer directory | the pack root |
//! | `TF_OUT_DIR` | a new directory for the scored files and the report | required |
//! | `TF_KV` | the KV profile of every execution: `f16`, `q8` or `q8k` | `f16` |
//! | `TF_TASKS` | task names, comma-separated, in request order | every task |
//! | `TF_PROBE` | `1`: probe the selected tasks, write `probe.json`, and stop (about 4 min per task at 4,096 tokens) | unset |
//! | `TF_TOKENS` | generated tokens per request | 4096 |
//! | `TF_REFERENCE_CHUNK` | prefill rows of the reference and the scorer, 1 to 8 | 8 |
//! | `TF_BASELINE_CHUNK` | prefill rows of the baseline, 1 to 8 | 4 |
//! | `TF_CANDIDATE` | `serve`, `fault` or generations-file paths, comma-separated; empty for none | `serve` |
//! | `TF_COMPARISON` | the directory of an earlier run whose `reference.json` and `baseline.json` this run reuses | unset |
//! | `TF_COMPARISON_SHA256` | `<reference>,<baseline>`: the digests those files must have, as the earlier run printed them | unset |
//! | `TF_AUDIT_CAP` | the most events audited per execution, `fault` excepted | 200, SpecPilot's count |
//! | `TF_WRITE_GENERATIONS` | write this build's serving generations to this new file and stop | unset |
//!
//! ```text
//! FLINT_QUAL_MODEL=<PACK> TF_OUT_DIR=<NEW_DIR> TF_CANDIDATE=fault,serve \
//! cargo test --release --features native-cuda --lib \
//!   family::flint::cuda_executor::token_forcing_tests::token_forcing_qualification \
//!   -- --ignored --exact --nocapture
//! ```
//!
//! The run writes `reference.json` and `baseline.json`, then per candidate `<name>.json` (its
//! scored outputs) and `report-<name>.json`, where the name is `serve`, `fault` or
//! `file<index>`. `family::token_forcing::run::entry::token_forcing_verdict` recomputes a
//! verdict from three scored files on any host. A `fault` that is not rejected for its own
//! mismatches after 2,048 tokens stops the run before the next candidate.
//!
//! The reference and the baseline take four of the six passes of a one-candidate run. With
//! `TF_COMPARISON`, a later run of the same test binary judges its candidates against the
//! files of an earlier one, so a qualification splits into leases of under an hour: the
//! comparison alone (`TF_CANDIDATE=`), then `fault`, then `serve`. The scorer of the later run
//! must equal the one recorded in the files, executable digest included. Every run prints a
//! `TOKEN_FORCING comparison` line with the files' digests, the calibration and the baseline's
//! counts; a later run given those digests in `TF_COMPARISON_SHA256` refuses other files, which
//! ties the leases of one qualification together.

use super::super::{factory, park_identity::pack_fingerprint};
use super::*;
use crate::family::token_forcing::{
    gate::{ExecutionIdentity, GateConfig, Generation, Report, Verdict, ids_sha256},
    run::{
        Candidate, Comparison, Identities, Outputs, RunSettings, input_digests, read_scored,
        write_new_json,
    },
    score::{ForcedModel, LogitBiasFault, TokenRecord, greedy_generate},
};
use crate::tokenizer::{IncrementalDecoder, PromptMessage, Role, Thinking, TokenizerBundle};
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
    tasks: Vec<String>,
    probe: bool,
    tokens: usize,
    reference_chunk: u32,
    baseline_chunk: u32,
    candidates: Vec<String>,
    audit_cap: usize,
    write_generations: Option<PathBuf>,
    comparison: Option<PathBuf>,
    comparison_sha256: Option<String>,
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
        let list = |name: &str| -> Vec<String> {
            std::env::var(name)
                .unwrap_or_default()
                .split(',')
                .map(|item| item.trim().to_owned())
                .filter(|item| !item.is_empty())
                .collect()
        };
        Self {
            tokenizer: std::env::var_os("FLINT_QUAL_TOKENIZER")
                .map_or_else(|| root.clone(), Into::into),
            out: std::env::var_os("TF_OUT_DIR")
                .expect("TF_OUT_DIR must name a new directory")
                .into(),
            kv: std::env::var("TF_KV").unwrap_or_else(|_| "f16".to_owned()),
            tasks: list("TF_TASKS"),
            probe: match std::env::var("TF_PROBE").as_deref() {
                Err(_) | Ok("") => false,
                Ok("1") => true,
                Ok(other) => panic!("TF_PROBE must be 1 or unset, not {other}"),
            },
            tokens: number("TF_TOKENS", 4096),
            reference_chunk: chunk("TF_REFERENCE_CHUNK", 8),
            baseline_chunk: chunk("TF_BASELINE_CHUNK", 4),
            candidates: std::env::var("TF_CANDIDATE")
                .unwrap_or_else(|_| "serve".to_owned())
                .split(',')
                .map(|name| name.trim().to_owned())
                .filter(|name| !name.is_empty())
                .collect(),
            audit_cap: number("TF_AUDIT_CAP", 200),
            write_generations: std::env::var_os("TF_WRITE_GENERATIONS").map(Into::into),
            comparison: std::env::var_os("TF_COMPARISON").map(Into::into),
            comparison_sha256: std::env::var("TF_COMPARISON_SHA256").ok(),
            root,
        }
    }

    fn context_limit(&self) -> u32 {
        let rows = (PROMPT_ROOM + self.tokens + 64).next_multiple_of(1024);
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

/// The longest rendered prompt. The context limit leaves this room before the outputs.
const PROMPT_ROOM: usize = 1024;

/// One committed task: a user message that asks for a long answer.
#[derive(Deserialize)]
struct Task {
    name: String,
    user: String,
}

/// The tasks `TF_TASKS` names, in its order, or every task, each rendered by the pack's chat
/// template as one user turn followed by the opening of the assistant's turn.
///
/// The gate refuses a repeated prompt; checking here stops the run before any generation.
fn prompts(plan: &Plan, bundle: &TokenizerBundle) -> Vec<(String, Vec<u32>)> {
    let tasks: Vec<Task> = serde_json::from_str(include_str!("testdata/token-forcing-tasks.json"))
        .expect("the committed task file");
    let mut seen = std::collections::BTreeSet::new();
    for task in &tasks {
        assert!(
            seen.insert(&task.name),
            "the task file repeats {}",
            task.name
        );
    }
    let names: Vec<&str> = if plan.tasks.is_empty() {
        tasks.iter().map(|task| task.name.as_str()).collect()
    } else {
        plan.tasks.iter().map(String::as_str).collect()
    };
    let mut selected = std::collections::BTreeSet::new();
    for name in &names {
        assert!(selected.insert(*name), "TF_TASKS names {name} twice");
    }
    let prompts: Vec<(String, Vec<u32>)> = names
        .into_iter()
        .map(|name| {
            let task = tasks
                .iter()
                .find(|task| task.name == name)
                .unwrap_or_else(|| panic!("TF_TASKS names {name}, which is not a task"));
            let message = PromptMessage {
                role: Role::User,
                content: Some(task.user.clone()),
                reasoning_content: None,
                name: None,
                tool_call_id: None,
                tool_calls: Vec::new(),
            };
            let ids = bundle
                .encode_prompt(&[message], &[], Thinking::Default)
                .expect("the task renders");
            assert!(
                ids.len() <= PROMPT_ROOM,
                "task {name} renders to {} ids, more than {PROMPT_ROOM}",
                ids.len()
            );
            (name.to_owned(), ids)
        })
        .collect();
    let distinct: std::collections::BTreeSet<_> = prompts
        .iter()
        .map(|(_, prompt)| ids_sha256(prompt))
        .collect();
    assert_eq!(
        distinct.len(),
        prompts.len(),
        "two tasks render to one prompt"
    );
    prompts
}

/// One task's reference output, read for what the fault self-test needs from the fault's
/// start up to the first end id: tokens that the fault changes by more than any event
/// threshold.
#[derive(Serialize)]
struct ProbeRow {
    task: String,
    prompt_sha256: String,
    prompt_tokens: usize,
    /// The first output position whose argmax ends generation for the tokenizer. The run does
    /// not stop there (see the module notes), so later tokens continue past the answer.
    first_end: Option<usize>,
    late_tokens: usize,
    late_distinct: usize,
    /// Late positions where the top two logits lie closer than the fault's bias.
    late_close: usize,
    /// Late positions where the fault's bias changes the argmax.
    fault_flips: usize,
    /// Flips whose margin exceeds `EVENT_FLOOR`.
    fault_events: usize,
}

/// The probe's file: what ran it, then one row per task in request order.
#[derive(Serialize)]
struct ProbeFile {
    system: ExecutionIdentity,
    tokens: usize,
    event_floor: f64,
    rows: Vec<ProbeRow>,
}

/// Twice the largest delta the gate accepts: a fault flip with a larger margin is an event
/// under every calibration that qualifies.
const EVENT_FLOOR: f64 = 2.0 * GateConfig::QUALIFICATION.max_delta;

/// Generates one reference output on the eager path, as `greedy_generate` does, and reads
/// every row of it for the probe.
fn probe(
    model: &mut dyn ForcedModel,
    (name, prompt): &(String, Vec<u32>),
    tokens: usize,
    bundle: &TokenizerBundle,
) -> ProbeRow {
    let vocab = bundle.vocab_size() as usize;
    let fault = LogitBiasFault::specpilot(());
    let mut out = ProbeRow {
        task: name.clone(),
        prompt_sha256: ids_sha256(prompt),
        prompt_tokens: prompt.len(),
        first_end: None,
        late_tokens: 0,
        late_distinct: 0,
        late_close: 0,
        fault_flips: 0,
        fault_events: 0,
    };
    let mut late = std::collections::BTreeSet::new();
    model.reset().expect("probe reset");
    let mut row = model.prefill_logits(prompt).expect("probe prefill");
    for position in 0..tokens {
        let record = TokenRecord::from_logits(position, &row, 0, vocab).expect("a probe row");
        let token = record.argmax;
        if out.first_end.is_none() && bundle.eos_token_ids().contains(&token) {
            out.first_end = Some(position);
        }
        if position >= fault.from && out.first_end.is_none() {
            late.insert(token);
            out.late_tokens += 1;
            out.late_close += usize::from(record.top2_gap < f64::from(fault.bias));
            let departure = fault
                .departure_at(position, &row, vocab)
                .expect("a biased probe row");
            if let Some(departure) = departure {
                out.fault_flips += 1;
                out.fault_events += usize::from(departure.margin() > EVENT_FLOOR);
            }
        }
        if position + 1 < tokens {
            row = model.step_logits(token).expect("probe step");
        }
    }
    out.late_distinct = late.len();
    out
}

/// Probes every selected task on the reference executor and writes `probe.json`.
fn run_probe(plan: &Plan, bundle: &TokenizerBundle, prompts: &[(String, Vec<u32>)]) {
    std::fs::create_dir(&plan.out).expect("TF_OUT_DIR must not exist yet");
    let (mut engine, mut session) = plan.open(plan.reference_chunk, bundle);
    let mut model = Forced {
        engine: &mut engine,
        session: &mut session,
    };
    let rows: Vec<ProbeRow> = prompts
        .iter()
        .map(|prompt| {
            let row = probe(&mut model, prompt, plan.tokens, bundle);
            println!(
                "TOKEN_FORCING probe task={} prompt_tokens={} first_end={:?} late_tokens={} \
                 late_distinct={} late_close={} fault_flips={} fault_events={}",
                row.task,
                row.prompt_tokens,
                row.first_end,
                row.late_tokens,
                row.late_distinct,
                row.late_close,
                row.fault_flips,
                row.fault_events
            );
            row
        })
        .collect();
    let file = ProbeFile {
        system: plan.identity("probe", plan.reference_chunk, "eager-greedy"),
        tokens: plan.tokens,
        event_floor: EVENT_FLOOR,
        rows,
    };
    let path = plan.out.join("probe.json");
    write_new_json(&path, &file).expect("the probe file");
    println!("TOKEN_FORCING probe={}", path.display());
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

/// What a candidate run must show. The fault must be rejected for its own mismatches, all
/// after 2,048 tokens; a lossless candidate must pass with every event audited.
fn check_outcome(name: &str, report: &Report, path: &std::path::Path) {
    if name == "fault" {
        assert!(
            matches!(report.verdict, Verdict::Reject { .. }),
            "the injected fault was not rejected; see {}",
            path.display()
        );
        let early: u64 = report.candidate.depth[..2]
            .iter()
            .map(|bin| bin.mismatches)
            .sum();
        assert_eq!(
            early,
            0,
            "the fault candidate mismatched before 2,048 tokens; see {}",
            path.display()
        );
        return;
    }
    assert_eq!(report.verdict, Verdict::Pass, "see {}", path.display());
    assert_eq!(
        report.candidate.unaudited_events,
        0,
        "see {}",
        path.display()
    );
}

/// What every part of one run shares.
struct Setup<'a> {
    plan: &'a Plan,
    bundle: &'a TokenizerBundle,
    prompts: &'a [(String, Vec<u32>)],
    identities: Identities,
    settings: RunSettings,
}

type Engine = (FlintExecutor, FlintSession);

/// Generates the baseline and the reference, scores and calibrates them, and writes their
/// files. Returns the comparison, the files' digests and the scorer's engine.
fn fresh_comparison(setup: &Setup<'_>) -> (Comparison, [String; 2], Engine) {
    let (plan, vocab) = (setup.plan, setup.settings.vocab);
    let baseline = {
        let (mut engine, mut session) = plan.open(plan.baseline_chunk, setup.bundle);
        let mut model = Forced {
            engine: &mut engine,
            session: &mut session,
        };
        generate_with(&mut model, setup.prompts, plan.tokens, vocab)
    };
    let (mut engine, mut session) = plan.open(plan.reference_chunk, setup.bundle);
    let mut scorer = Forced {
        engine: &mut engine,
        session: &mut session,
    };
    let reference = generate_with(&mut scorer, setup.prompts, plan.tokens, vocab);
    let outputs = Outputs {
        reference: &reference,
        baseline: &baseline,
    };
    let comparison = Comparison::prepare(
        (&mut scorer, None),
        &setup.identities,
        outputs,
        setup.settings,
    )
    .expect("the reference and the baseline calibrate");
    let write = |role: &str, scored| {
        write_new_json(&plan.out.join(format!("{role}.json")), scored).expect("a comparison file")
    };
    let digests = [
        write("reference", &comparison.reference),
        write("baseline", &comparison.baseline),
    ];
    (comparison, digests, (engine, session))
}

/// Reads the comparison an earlier run of this binary wrote to `directory`.
fn loaded_comparison(
    setup: &Setup<'_>,
    directory: &std::path::Path,
) -> (Comparison, [String; 2], Engine) {
    let (reference, reference_sha256) =
        read_scored(&directory.join("reference.json")).expect("the earlier reference file");
    let (baseline, baseline_sha256) =
        read_scored(&directory.join("baseline.json")).expect("the earlier baseline file");
    let comparison = Comparison::from_scored(
        reference,
        baseline,
        &setup.identities.scorer,
        setup.settings,
    )
    .expect("the earlier comparison under this scorer");
    let engine = setup.plan.open(setup.plan.reference_chunk, setup.bundle);
    (comparison, [reference_sha256, baseline_sha256], engine)
}

/// The comparison as one line: the digests that tie later leases to these files, the
/// calibration, and the baseline counts that the rules compare every candidate with.
fn comparison_line(comparison: &Comparison, [reference, baseline]: &[String; 2]) -> String {
    let records = || comparison.baseline.requests.iter().flat_map(|r| &r.records);
    let tokens = records().count();
    let mismatches = records().filter(|record| !record.matched()).count();
    let reference_mismatches = comparison
        .reference
        .requests
        .iter()
        .flat_map(|request| &request.records)
        .filter(|record| !record.matched())
        .count();
    let mut events = 0;
    let mut unaudited = 0;
    for request in &comparison.baseline.requests {
        for record in &request.records {
            if record.margin() > comparison.threshold() {
                events += 1;
                let audited = request.audits.iter().any(|a| a.position == record.position);
                unaudited += usize::from(!audited);
            }
        }
    }
    let calibration = &comparison.calibration;
    format!(
        "TOKEN_FORCING comparison reference_sha256={reference} baseline_sha256={baseline} \
         delta={} largest_margin={} departures={} reference_mismatches={reference_mismatches} \
         baseline_mismatches={mismatches}/{tokens} baseline_events={events} \
         baseline_unaudited={unaudited}",
        calibration.delta,
        calibration.largest_margin,
        calibration.departures.len()
    )
}

/// Generates one candidate's outputs: its file name, identity, generations and audit cap.
fn candidate_outputs(
    setup: &Setup<'_>,
    (engine, session): (&mut FlintExecutor, &mut FlintSession),
    (index, kind): (usize, &str),
) -> (String, ExecutionIdentity, Vec<Generation>, usize) {
    let plan = setup.plan;
    match kind {
        "serve" => (
            "serve".to_owned(),
            plan.identity("candidate-serve", plan.reference_chunk, "serving-decode"),
            serve((engine, session), setup.bundle, setup.prompts, plan.tokens),
            plan.audit_cap,
        ),
        "fault" => {
            let mut model = LogitBiasFault::specpilot(Forced { engine, session });
            (
                "fault".to_owned(),
                plan.identity(
                    "candidate-fault",
                    plan.reference_chunk,
                    "eager-greedy-bias-fault",
                ),
                generate_with(&mut model, setup.prompts, plan.tokens, setup.settings.vocab),
                0,
            )
        }
        path => {
            let (identity, generations) =
                read_generations(std::path::Path::new(path), setup.prompts);
            (
                format!("file{index}"),
                identity,
                generations,
                plan.audit_cap,
            )
        }
    }
}

#[test]
#[ignore = "needs the FLINT_QUAL_MODEL EXL3 pack and an exclusive GPU lease"]
fn token_forcing_qualification() {
    let plan = Plan::from_env();
    let bundle = factory::load_tokenizer(&plan.tokenizer).expect("the qualification tokenizer");
    let vocab = bundle.vocab_size() as usize;
    let prompts = prompts(&plan, &bundle);
    if plan.probe {
        assert!(
            plan.write_generations.is_none(),
            "TF_PROBE and TF_WRITE_GENERATIONS are separate runs"
        );
        run_probe(&plan, &bundle, &prompts);
        return;
    }
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
    let setup = Setup {
        plan: &plan,
        bundle: &bundle,
        prompts: &prompts,
        identities: Identities {
            scorer: plan.identity("scorer", plan.reference_chunk, "eager-teacher-forced"),
            reference: plan.identity("reference", plan.reference_chunk, "eager-greedy"),
            baseline: plan.identity("baseline", plan.baseline_chunk, "eager-greedy"),
        },
        settings: RunSettings {
            config: GateConfig::QUALIFICATION,
            vocab,
            audit_cap: plan.audit_cap,
        },
    };
    let (comparison, digests, (mut engine, mut session)) = match &plan.comparison {
        Some(directory) => loaded_comparison(&setup, directory),
        None => {
            assert!(
                plan.comparison_sha256.is_none(),
                "TF_COMPARISON_SHA256 names the files of TF_COMPARISON, which is unset"
            );
            fresh_comparison(&setup)
        }
    };
    println!("{}", comparison_line(&comparison, &digests));
    if let Some(expected) = &plan.comparison_sha256 {
        assert_eq!(
            *expected,
            digests.join(","),
            "the comparison files are not the ones TF_COMPARISON_SHA256 names"
        );
    }
    if plan.candidates.is_empty() {
        println!("TOKEN_FORCING candidates=none");
    }
    let [reference_sha256, baseline_sha256] = digests;
    for (index, kind) in plan.candidates.iter().enumerate() {
        let (name, identity, generations, audit_cap) =
            candidate_outputs(&setup, (&mut engine, &mut session), (index, kind));
        let mut scorer = Forced {
            engine: &mut engine,
            session: &mut session,
        };
        let candidate = Candidate {
            identity: &identity,
            generations: &generations,
            audit_cap,
        };
        let mut run = comparison
            .judge((&mut scorer, None), candidate)
            .expect("the gate runs");
        let candidate_sha256 =
            write_new_json(&plan.out.join(format!("{name}.json")), &run.candidate)
                .expect("a candidate file");
        run.report.inputs = input_digests([
            reference_sha256.clone(),
            baseline_sha256.clone(),
            candidate_sha256,
        ]);
        let report_path = plan.out.join(format!("report-{name}.json"));
        write_new_json(&report_path, &run.report).expect("the report");
        println!(
            "TOKEN_FORCING candidate={name} verdict={} delta={} report={}",
            serde_json::to_string(&run.report.verdict).expect("a verdict serializes"),
            run.report.calibration.delta,
            report_path.display()
        );
        check_outcome(&name, &run.report, &report_path);
    }
}
