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
//! turn through the pack's chat template, asking for a long coding, design or derivation
//! answer. Raw-text continuations of the teacher-forced gate's corpus were tried first and
//! cannot qualify anything: by 2,048 tokens every output had become a repetition loop (2 to 35
//! distinct ids, median top-two gap 8.6 nats), the injected fault changed no token, and the
//! self-test passed a faulty candidate. `TF_PROBE=1` measures that risk before a
//! qualification (`token_forcing::probe`): it generates the reference output of each selected
//! task, counts the fault's reach in it from the fault's start up to the first end id, and
//! writes `probe.json`. The counts are zero when `TF_TOKENS` is at most 2,048, and text after
//! an end id does not count, because the run continues past it (see above). A later run given
//! `TF_PROBE_FILE` judges no candidate unless the probe read exactly the outputs that its
//! reference scored, on the same build, pack and profile.
//!
//! Audits run with the scorer only: a second executor does not fit beside it on a 24 GB
//! card, so the baseline-model audit is absent from device reports. Each audit is one prefill of
//! the whole context up to its event, so the `fault` self-test is not audited: its events
//! count as unexplained, and auditing hundreds of them would take hours. A low `TF_AUDIT_CAP`
//! cannot help a faulty candidate pass either, for the same reason.
//!
//! A second opt-in test, `token_forcing_audit_path`, runs the device audit, which a
//! qualification need not reach: the serving candidate may have no events, and the fault
//! self-test is not audited. It runs a short comparison (`TF_TOKENS`, 512 by default) and a
//! candidate with the fault from output position 128, and audits up to `TF_AUDIT_CAP` of the
//! candidate's events. Its defaults are those of its passing device run (`AUDIT_PATH`), so it
//! runs as intended with no settings. It asserts that each audit is one prefill of the whole
//! context up to its event; that the re-read row gives the step path's argmax and margin
//! bitwise, because Flint's prefill is batch-invariant, so that no legal re-execution here
//! explains an event; and that the report counts the audited events as unexplained and the rest
//! as unaudited, and rejects. The cap must be at least 2, and the events must exceed it by at least five
//! unaudited ones (`MIN_UNAUDITED`), so that the unaudited-event accounting carries as many
//! events as the event rule needs to reject.
//!
//! | Variable | Meaning | Default |
//! | --- | --- | --- |
//! | `FLINT_QUAL_MODEL` | the EXL3 pack root | required |
//! | `FLINT_QUAL_TOKENIZER` | the tokenizer directory | the pack root |
//! | `TF_OUT_DIR` | a new directory for the scored files and the report | required |
//! | `TF_KV` | the KV profile of every execution: `f16`, `q8` or `q8k` | `f16` |
//! | `TF_TASKS` | task names, comma-separated, in request order | every task; `rate-limiter-design,probability-derivation` for `token_forcing_audit_path` |
//! | `TF_PROBE` | `1`: probe the selected tasks, write `probe.json`, and stop (about 80 s per task at 4,096 tokens) | unset |
//! | `TF_PROBE_FILE` | a `probe.json` that must have read this run's reference outputs; checked before any candidate | unset |
//! | `TF_TOKENS` | generated tokens per request | 4096; 512 for `token_forcing_audit_path` |
//! | `TF_REFERENCE_CHUNK` | prefill rows of the reference and the scorer, 1 to 8 | 8 |
//! | `TF_BASELINE_CHUNK` | prefill rows of the baseline, 1 to 8 | 4 |
//! | `TF_CANDIDATE` | `serve`, `fault` or generations-file paths, comma-separated; empty for none | `serve` |
//! | `TF_COMPARISON` | the directory of an earlier run whose `reference.json` and `baseline.json` this run reuses | unset |
//! | `TF_COMPARISON_SHA256` | `<reference>,<baseline>`: the digests those files must have, as the earlier run printed them | unset |
//! | `TF_AUDIT_CAP` | the most events audited per execution, `fault` excepted | 200, SpecPilot's count; 8 for `token_forcing_audit_path` |
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
    gate::{
        Audit, EXECUTABLE_SHA256, ExecutionIdentity, GENERATION_PATH, GateConfig, Generation,
        PREFILL_CHUNK_SIZE, Report, ScoredSystem, Verdict, ids_sha256,
    },
    probe::{EVENT_FLOOR, ProbeFile, ProbeRow, ProbeSettings, check_probe, probe_request},
    run::{
        Candidate, Comparison, Identities, Outputs, Run, RunSettings, input_digests, read_scored,
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
    probe_file: Option<PathBuf>,
}

/// What a test runs when the environment does not say.
#[derive(Clone, Copy, Debug)]
struct Defaults {
    /// `TF_TOKENS`.
    tokens: usize,
    /// `TF_AUDIT_CAP`.
    audit_cap: usize,
    /// `TF_TASKS`; empty for every task.
    tasks: &'static [&'static str],
}

/// The qualification: every task unless `TF_TASKS` names some (its driver names the four that
/// the probe selects), 4,096 tokens per request, and SpecPilot's audit count.
const QUALIFICATION: Defaults = Defaults {
    tokens: 4096,
    audit_cap: 200,
    tasks: &[],
};

/// The audit-path test, as its passing device run ran it: two tasks whose faulty outputs give
/// many more events than the cap (57 events, 8 audited, on the EXL3 4.0 pack).
const AUDIT_PATH: Defaults = Defaults {
    tokens: 512,
    audit_cap: 8,
    tasks: &["rate-limiter-design", "probability-derivation"],
};

impl Plan {
    /// The plan the environment sets, with `defaults` where it sets nothing.
    fn from_env(defaults: Defaults) -> Self {
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
            tasks: match list("TF_TASKS") {
                tasks if tasks.is_empty() => {
                    defaults.tasks.iter().map(|&task| task.to_owned()).collect()
                }
                tasks => tasks,
            },
            probe: match std::env::var("TF_PROBE").as_deref() {
                Err(_) | Ok("") => false,
                Ok("1") => true,
                Ok(other) => panic!("TF_PROBE must be 1 or unset, not {other}"),
            },
            tokens: number("TF_TOKENS", defaults.tokens),
            reference_chunk: chunk("TF_REFERENCE_CHUNK", 8),
            baseline_chunk: chunk("TF_BASELINE_CHUNK", 4),
            candidates: std::env::var("TF_CANDIDATE")
                .unwrap_or_else(|_| "serve".to_owned())
                .split(',')
                .map(|name| name.trim().to_owned())
                .filter(|name| !name.is_empty())
                .collect(),
            audit_cap: number("TF_AUDIT_CAP", defaults.audit_cap),
            write_generations: std::env::var_os("TF_WRITE_GENERATIONS").map(Into::into),
            comparison: std::env::var_os("TF_COMPARISON").map(Into::into),
            comparison_sha256: std::env::var("TF_COMPARISON_SHA256").ok(),
            probe_file: std::env::var_os("TF_PROBE_FILE")
                .filter(|path| !path.is_empty())
                .map(Into::into),
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
}

/// The identity of one execution of `engine`, with its settings read from the executor (see
/// [`effective_profile`]) and `path` naming how it generated.
fn identity(label: &str, engine: &FlintExecutor, path: &str) -> ExecutionIdentity {
    let fingerprint = pack_fingerprint(&engine.root).expect("the pack fingerprint");
    let mut profile = effective_profile(engine);
    profile.insert(GENERATION_PATH.to_owned(), path.to_owned());
    ExecutionIdentity {
        label: label.to_owned(),
        family: "flint-cuda".to_owned(),
        weights: hex(&fingerprint),
        weight_format: "exl3".to_owned(),
        profile,
    }
}

/// The scorer's identity: `engine`'s eager path reading each output teacher-forced.
fn scorer_identity(engine: &FlintExecutor) -> ExecutionIdentity {
    identity("scorer", engine, "eager-teacher-forced")
}

/// The settings `engine` runs with, read from the executor rather than from the plan or the
/// environment, so that a profile cannot claim settings the run did not use: the options it
/// was opened with, the effective selectors its numeric path binds (those of its prefix
/// fingerprint, with its own layer-major mode), and the resolved kernel switches of its load
/// line. `gate::check_builds` compares these entries: the baseline may differ from the scorer
/// in the prefill chunk size only, so a q8 baseline beside an f16 scorer is refused.
///
/// Left out on purpose: the expert-bank split, because the work reserve grows with the prefill
/// rows, so the bank can differ between a chunk-4 baseline and a chunk-8 scorer on one numeric
/// path, and recording it would make a legal baseline refuse itself; and the lazy allocation
/// return, which changes when a freed block may be reused, not what a kernel computes.
fn effective_profile(engine: &FlintExecutor) -> BTreeMap<String, String> {
    let options = &engine.options;
    let numeric_env = CudaPrefixEnv {
        layer_major: engine.layer_mode,
        ..frozen_prefix_env()
    };
    let switches = [
        resolved_switch_suffix(
            packed_pointwise_enabled(),
            grouped_gather_enabled(),
            resolved_q8_tiled(&options.kv_profile, q8_tiled_armed),
            kv_device_f16_armed(),
        ),
        fused_publish_suffix(fused_publish_armed()),
        engine.ctx.single_gather_suffix(),
        format!(
            " token_health={} device_picks={} layer_major_rows={:?}",
            token_health_armed(),
            device_picks_armed(),
            engine.layer_major
        ),
    ]
    .concat();
    let cpu_moe = format!(
        "offload_layers={} split_experts={} threads={}",
        options.cpu_moe_offload_layers, options.cpu_moe_split_experts, options.cpu_moe_threads
    );
    BTreeMap::from([
        ("kv_profile".to_owned(), options.kv_profile.clone()),
        (
            PREFILL_CHUNK_SIZE.to_owned(),
            engine.prefill_chunk_size.to_string(),
        ),
        ("context_limit".to_owned(), engine.context_limit.to_string()),
        ("cpu_moe".to_owned(), cpu_moe),
        ("numeric_env".to_owned(), format!("{numeric_env:?}")),
        (
            "kernel_switches".to_owned(),
            switches.trim_start().to_owned(),
        ),
        (EXECUTABLE_SHA256.to_owned(), executable_sha256()),
    ])
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

/// Probes every selected task on the reference executor and writes `probe.json`.
fn run_probe(plan: &Plan, bundle: &TokenizerBundle, prompts: &[(String, Vec<u32>)]) {
    std::fs::create_dir(&plan.out).expect("TF_OUT_DIR must not exist yet");
    let (mut engine, mut session) = plan.open(plan.reference_chunk, bundle);
    let mut model = Forced {
        engine: &mut engine,
        session: &mut session,
    };
    let fault = LogitBiasFault::specpilot(());
    let settings = ProbeSettings {
        fault: &fault,
        end_ids: bundle.eos_token_ids(),
        tokens: plan.tokens,
        vocab: bundle.vocab_size() as usize,
    };
    let rows: Vec<ProbeRow> = prompts
        .iter()
        .map(|(name, prompt)| {
            let row = probe_request(&mut model, (name, prompt), settings).expect("a probe");
            println!(
                "TOKEN_FORCING probe task={} prompt_tokens={} generated_sha256={} \
                 first_end={:?} late_tokens={} late_distinct={} late_close={} fault_flips={} \
                 fault_events={}",
                row.task,
                row.prompt_tokens,
                row.generated_sha256,
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
        system: identity("probe", &engine, "eager-greedy"),
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
    settings: RunSettings,
}

type Engine = (FlintExecutor, FlintSession);

/// Generates the baseline and the reference, scores and calibrates them, and writes their
/// files. Returns the comparison, the files' digests and the scorer's engine.
fn fresh_comparison(setup: &Setup<'_>) -> (Comparison, [String; 2], Engine) {
    let (plan, vocab) = (setup.plan, setup.settings.vocab);
    let (baseline, baseline_identity) = {
        let (mut engine, mut session) = plan.open(plan.baseline_chunk, setup.bundle);
        let system = identity("baseline", &engine, "eager-greedy");
        let mut model = Forced {
            engine: &mut engine,
            session: &mut session,
        };
        let generations = generate_with(&mut model, setup.prompts, plan.tokens, vocab);
        (generations, system)
    };
    let (mut engine, mut session) = plan.open(plan.reference_chunk, setup.bundle);
    let identities = Identities {
        scorer: scorer_identity(&engine),
        reference: identity("reference", &engine, "eager-greedy"),
        baseline: baseline_identity,
    };
    let mut scorer = Forced {
        engine: &mut engine,
        session: &mut session,
    };
    let reference = generate_with(&mut scorer, setup.prompts, plan.tokens, vocab);
    let outputs = Outputs {
        reference: &reference,
        baseline: &baseline,
    };
    let comparison = Comparison::prepare((&mut scorer, None), &identities, outputs, setup.settings)
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
    let engine = setup.plan.open(setup.plan.reference_chunk, setup.bundle);
    let comparison = Comparison::from_scored(
        reference,
        baseline,
        &scorer_identity(&engine.0),
        setup.settings,
    )
    .expect("the earlier comparison under this scorer");
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
            identity("candidate-serve", engine, "serving-decode"),
            serve((engine, session), setup.bundle, setup.prompts, plan.tokens),
            plan.audit_cap,
        ),
        "fault" => {
            let system = identity("candidate-fault", engine, "eager-greedy-bias-fault");
            let mut model = LogitBiasFault::specpilot(Forced { engine, session });
            (
                "fault".to_owned(),
                system,
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

/// The plan, the prompts and the gate settings of a run. The identities the files record are
/// read from each executor once it is open.
fn setup<'a>(
    plan: &'a Plan,
    bundle: &'a TokenizerBundle,
    prompts: &'a [(String, Vec<u32>)],
) -> Setup<'a> {
    Setup {
        plan,
        bundle,
        prompts,
        settings: RunSettings {
            config: GateConfig::QUALIFICATION,
            vocab: bundle.vocab_size() as usize,
            audit_cap: plan.audit_cap,
        },
    }
}

/// Stops the run unless the probe file at `path` read the outputs this run's reference scored.
fn check_probe_file(path: &std::path::Path, reference: &ScoredSystem) {
    let bytes = std::fs::read(path).expect("the TF_PROBE_FILE probe");
    let probe: ProbeFile = serde_json::from_slice(&bytes).expect("probe JSON");
    check_probe(&probe, reference).unwrap_or_else(|error| panic!("{}: {error:?}", path.display()));
    println!(
        "TOKEN_FORCING probe_check=ok probe={} probe_sha256={:x}",
        path.display(),
        Sha256::digest(&bytes)
    );
}

/// Writes one candidate's scored file and report, tied to the comparison files by digest, and
/// returns the report's path.
fn write_run(
    plan: &Plan,
    name: &str,
    run: &mut Run,
    [reference, baseline]: &[String; 2],
) -> PathBuf {
    let candidate = write_new_json(&plan.out.join(format!("{name}.json")), &run.candidate)
        .expect("a candidate file");
    run.report.inputs = input_digests([reference.clone(), baseline.clone(), candidate]);
    let report_path = plan.out.join(format!("report-{name}.json"));
    write_new_json(&report_path, &run.report).expect("the report");
    println!(
        "TOKEN_FORCING candidate={name} verdict={} delta={} report={}",
        serde_json::to_string(&run.report.verdict).expect("a verdict serializes"),
        run.report.calibration.delta,
        report_path.display()
    );
    report_path
}

#[test]
#[ignore = "needs the FLINT_QUAL_MODEL EXL3 pack and an exclusive GPU lease"]
fn token_forcing_qualification() {
    let plan = Plan::from_env(QUALIFICATION);
    let bundle = factory::load_tokenizer(&plan.tokenizer).expect("the qualification tokenizer");
    let prompts = prompts(&plan, &bundle);
    if plan.probe {
        assert!(
            plan.write_generations.is_none()
                && plan.comparison.is_none()
                && plan.probe_file.is_none(),
            "TF_PROBE runs alone: no TF_WRITE_GENERATIONS, TF_COMPARISON or TF_PROBE_FILE"
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
        let system = identity("candidate-serve", &engine, "serving-decode");
        write_new_json(path, &GenerationsFile { system, requests }).expect("generations file");
        println!("TOKEN_FORCING generations={}", path.display());
        return;
    }
    std::fs::create_dir(&plan.out).expect("TF_OUT_DIR must not exist yet");
    let setup = setup(&plan, &bundle, &prompts);
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
    if let Some(path) = &plan.probe_file {
        check_probe_file(path, &comparison.reference);
    }
    if plan.candidates.is_empty() {
        println!("TOKEN_FORCING candidates=none");
    }
    for (index, kind) in plan.candidates.iter().enumerate() {
        let (name, system, generations, audit_cap) =
            candidate_outputs(&setup, (&mut engine, &mut session), (index, kind));
        let mut scorer = Forced {
            engine: &mut engine,
            session: &mut session,
        };
        let candidate = Candidate {
            identity: &system,
            generations: &generations,
            audit_cap,
        };
        let mut run = comparison
            .judge((&mut scorer, None), candidate)
            .expect("the gate runs");
        let report_path = write_run(&plan, &name, &mut run, &digests);
        check_outcome(&name, &run.report, &report_path);
    }
}

/// The output position where the audit-path test's fault starts: early, so that a short run
/// has events to audit.
const AUDIT_FAULT_FROM: usize = 128;

/// The fewest unaudited events the audit-path test accepts: five unexplained events are the
/// fewest that the event rule rejects on at equal exposure and no baseline events (four give
/// p = 1/16).
const MIN_UNAUDITED: usize = 5;

/// A model that records the context length of every prefill it runs.
struct Prefills<'a> {
    inner: Forced<'a>,
    contexts: Vec<usize>,
}

impl ForcedModel for Prefills<'_> {
    fn reset(&mut self) -> Result<()> {
        self.inner.reset()
    }

    fn prefill_logits(&mut self, context: &[u32]) -> Result<Vec<f32>> {
        self.contexts.push(context.len());
        self.inner.prefill_logits(context)
    }

    fn step_logits(&mut self, token: u32) -> Result<Vec<f32>> {
        self.inner.step_logits(token)
    }
}

/// What the audit-path test asserts about one judged faulty candidate; see the module notes.
/// `contexts` are the prefills the scorer ran while judging it.
fn check_audit_path(run: &Run, generations: &[Generation], contexts: &[usize], cap: usize) {
    let threshold = run.report.event_rule.threshold;
    let scored = &run.candidate;
    // The events in request and position order, the order in which the audit walks them.
    let events: Vec<(usize, &TokenRecord)> = scored
        .requests
        .iter()
        .enumerate()
        .flat_map(|(index, request)| {
            let above = request.records.iter().filter(|r| r.margin() > threshold);
            above.map(move |record| (index, record))
        })
        .collect();
    let audited = events.len().min(cap);
    assert!(
        audited >= 2 && events.len() > cap,
        "the fault made {} events; the test needs at least 2 audited and more than the cap {cap}",
        events.len()
    );
    assert!(
        events.len() - audited >= MIN_UNAUDITED,
        "the fault made {} events and {audited} were audited; the test needs at least \
         {MIN_UNAUDITED} unaudited events, and so at least {MIN_UNAUDITED} events",
        events.len()
    );
    let records = scored.requests.iter().flat_map(|request| &request.records);
    let mut early = records.filter(|record| record.position < AUDIT_FAULT_FROM);
    assert!(
        early.all(TokenRecord::matched),
        "a mismatch before the fault's start"
    );
    let audits: Vec<(usize, &Audit)> = scored
        .requests
        .iter()
        .enumerate()
        .flat_map(|(index, request)| request.audits.iter().map(move |audit| (index, audit)))
        .collect();
    assert_eq!(audits.len(), audited, "one audit per event up to the cap");
    // Scoring reads each prompt once; then each audit is one prefill of its whole context.
    let mut expected: Vec<usize> = generations.iter().map(|g| g.prompt.len()).collect();
    expected.extend(
        events[..audited]
            .iter()
            .map(|&(index, record)| generations[index].prompt.len() + record.position),
    );
    assert_eq!(contexts, expected, "the prefills the scorer ran");
    for (&(index, record), &(audit_index, audit)) in events.iter().zip(&audits) {
        println!(
            "TOKEN_FORCING audit request={} position={} context={} generated={} \
             step_argmax={} step_margin={} reread_argmax={} reread_margin={}",
            scored.requests[index].name,
            record.position,
            generations[index].prompt.len() + record.position,
            record.generated,
            record.argmax,
            record.margin(),
            audit.prefill.argmax,
            audit.prefill.margin
        );
        assert_eq!((audit_index, audit.position), (index, record.position));
        assert!(audit.baseline.is_none(), "the device has no baseline model");
    }
    for (&(_, record), &(_, audit)) in events.iter().zip(&audits) {
        assert_eq!(
            (audit.prefill.argmax, audit.prefill.margin.to_bits()),
            (record.argmax, record.margin().to_bits()),
            "position {}: one prefill of the whole context read another row than the step path",
            record.position
        );
    }
    let summary = &run.report.candidate;
    let counts = (
        summary.events,
        summary.explained_events,
        summary.unexplained_events,
        summary.unaudited_events,
    );
    let (events, audited) = (events.len() as u64, audited as u64);
    assert_eq!(counts, (events, 0, audited, events - audited));
    assert_eq!(run.report.event_rule.candidate_unexplained, events);
    assert!(matches!(run.report.verdict, Verdict::Reject { .. }));
}

#[test]
#[ignore = "needs the FLINT_QUAL_MODEL EXL3 pack and a GPU lease"]
fn token_forcing_audit_path() {
    let plan = Plan::from_env(AUDIT_PATH);
    assert!(
        plan.tokens > AUDIT_FAULT_FROM,
        "TF_TOKENS must pass the fault's start"
    );
    assert!(plan.audit_cap >= 2, "TF_AUDIT_CAP must allow two audits");
    let bundle = factory::load_tokenizer(&plan.tokenizer).expect("the qualification tokenizer");
    let prompts = prompts(&plan, &bundle);
    std::fs::create_dir(&plan.out).expect("TF_OUT_DIR must not exist yet");
    let setup = setup(&plan, &bundle, &prompts);
    let (comparison, digests, (mut engine, mut session)) = fresh_comparison(&setup);
    println!("{}", comparison_line(&comparison, &digests));
    let generations = {
        let forced = Forced {
            engine: &mut engine,
            session: &mut session,
        };
        let mut model = LogitBiasFault::starting_at(forced, AUDIT_FAULT_FROM);
        generate_with(&mut model, &prompts, plan.tokens, setup.settings.vocab)
    };
    let path = format!("eager-greedy-bias-fault-from-{AUDIT_FAULT_FROM}");
    let system = identity("candidate-fault", &engine, &path);
    let mut scorer = Prefills {
        inner: Forced {
            engine: &mut engine,
            session: &mut session,
        },
        contexts: Vec::new(),
    };
    let candidate = Candidate {
        identity: &system,
        generations: &generations,
        audit_cap: plan.audit_cap,
    };
    let mut run = comparison
        .judge((&mut scorer, None), candidate)
        .expect("the gate runs");
    write_run(&plan, "fault", &mut run, &digests);
    check_audit_path(&run, &generations, &scorer.contexts, plan.audit_cap);
}
