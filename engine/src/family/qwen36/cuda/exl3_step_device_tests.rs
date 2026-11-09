//! The EXL3 decode step against the real pack, guarding the arena lifetimes of the step.
//!
//! The step plan reuses dead storage. A slot whose lifetime is shorter than its real use is
//! overwritten by a later producer and the step returns garbage tokens, which the plan-level tests
//! see only as a lifetime comparison. This test runs the same greedy decode twice over the loaded
//! pack: once on the planned arena and once on an arena in which no two slots share bytes. It
//! needs no recorded output, because the two runs must agree token for token and the logits rows
//! must agree to float tolerance.
//!
//! `FIFI_EXL3_CHECKPOINT` names the pack directory. `FIFI_EXL3_GREEDY_IDS`, when set, is a
//! comma-separated list of token ids the planned run must begin with; the test prints the ids it
//! decoded so a good run can be recorded.
//!
//! Recorded on the 4.00 bpw pack (version 0.0.29), both arenas, relative L2 0 on all 16 rows. The
//! ids spell `<think>\nHere's a thinking process:\n\n1.  **Analyze User`:
//! `FIFI_EXL3_GREEDY_IDS=248068,198,8160,579,264,7047,1817,25,271,16,13,220,2972,2014,53983,2570`

use std::path::Path;

use super::{
    common_tests::{dims as step_dims, inventory},
    dims::{KvMode, StepOptions},
    exl3_load::load_exl3_dir,
    exl3_load_device_tests::relative_l2,
    layer_table::LayerTable,
    mixer_geometry::MixerGeometry,
    native_device_rig::device_present,
    resources::{GlobalNames, STEP_ROWS, StepResources},
    rules::QwenRules,
    run_report::CaptureOutcome,
    step::{CaptureConfig, QwenResidentStep},
    step_plan::{KernelScratch, StepPlan},
};
use crate::{
    EngError,
    family::{
        gguf_model::Role,
        project::cuda::{CudaProjector, CudaWeight, Reserve, WeightLayout},
    },
    kernels::cuda::{ArenaTensor, CudaContext},
};

/// The sampler vocabulary, the padded head width and the stop ids of the pinned model.
pub(super) const SAMPLER_VOCAB: usize = 248_077;
pub(super) const HEAD_ROWS: usize = 248_320;
pub(super) const EOS: [u32; 2] = [248_046, 248_044];
/// The chat-templated user turn "Explain in three sentences what a hash map is and why lookups
/// are fast." ending at the assistant header, in the ids of the model's tokenizer.
pub(super) const PROMPT: [u32; 25] = [
    248045, 846, 198, 814, 20139, 303, 2250, 22157, 1092, 264, 5010, 2336, 369, 321, 3069, 1353,
    8339, 513, 4778, 13, 248046, 198, 248045, 74455, 198,
];
pub(super) const GENERATED: usize = 16;
/// The aliasing bug gave rows unrelated to the true ones (relative L2 near 1); float summation
/// order may differ between the two arenas only in the last bits.
const ROW_BOUND: f64 = 1e-3;

struct Run {
    tokens: Vec<u32>,
    rows: Vec<Vec<u8>>,
}

struct Pair {
    planned: Run,
    wide: Run,
    planned_arena_bytes: usize,
    wide_arena_bytes: usize,
}

pub(super) fn options() -> StepOptions {
    StepOptions {
        kv_mode: KvMode::Int8G32,
        context_tokens: PROMPT.len() + GENERATED + 1,
        delta_slots: 1,
        forked_experts: false,
        tiered: None,
        layout: WeightLayout::Exl3,
    }
}

/// The resources of the step with every slot widened to the whole step, so no two share bytes.
fn unaliased_resources(
    context: &CudaContext,
    options: &StepOptions,
) -> Result<StepResources, EngError> {
    let dims = step_dims();
    let inventory = inventory();
    let scratch = KernelScratch::for_dims(&dims, STEP_ROWS, options.kv_mode)?;
    let plan = StepPlan::new(&dims, STEP_ROWS, options, &scratch)?;
    let planned = plan.arena_tensors();
    let end = planned.iter().map(|tensor| tensor.last).max().unwrap_or(0);
    let wide: Vec<ArenaTensor> = planned
        .into_iter()
        .map(|tensor| ArenaTensor {
            first: 0,
            last: end,
            ..tensor
        })
        .collect();
    Ok(StepResources {
        arena: context.arena(&wide)?,
        table: LayerTable::new(&dims, &inventory)?,
        geometry: MixerGeometry::new(&dims)?,
        names: GlobalNames::new(&inventory)?,
        dims,
        plan,
        paged: None,
        sparse: None,
        krow: None,
    })
}

fn decode(step: &mut QwenResidentStep<'_>) -> Result<Run, EngError> {
    let Some((last, lead)) = PROMPT.split_last() else {
        return Err(EngError::BadRequest("EXL3 decode: empty prompt".into()));
    };
    for &id in lead {
        step.run_token(id, false)?;
    }
    let mut next = step.run_token(*last, true)?;
    let mut run = Run {
        tokens: Vec::new(),
        rows: Vec::new(),
    };
    for _ in 0..GENERATED {
        run.tokens.push(next);
        run.rows.push(step.row_bytes().to_vec());
        next = step.run_token(next, true)?;
    }
    Ok(run)
}

/// Device phase: fallible operations only; the caller asserts after every owner has dropped.
fn both_runs(dir: &Path) -> Result<Pair, EngError> {
    let projector = CudaProjector::new()?;
    let weights = load_exl3_dir(&projector, dir, Reserve::default())?;
    let context = projector.context();
    let options = options();
    let rules = || QwenRules::new(EOS.to_vec(), SAMPLER_VOCAB, HEAD_ROWS);
    let (planned, planned_arena_bytes) = {
        let resources = StepResources::new(context, step_dims(), &inventory(), &options)?;
        let mut step = QwenResidentStep::new(context, &resources, &weights, &options, rules()?)?;
        (decode(&mut step)?, resources.plan.arena_bytes())
    };
    let (wide, wide_arena_bytes) = {
        let resources = unaliased_resources(context, &options)?;
        let mut step = QwenResidentStep::new(context, &resources, &weights, &options, rules()?)?;
        (decode(&mut step)?, resources.plan.unaliased_bytes())
    };
    Ok(Pair {
        planned,
        wide,
        planned_arena_bytes,
        wide_arena_bytes,
    })
}

pub(super) fn row_values(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

#[test]
#[ignore = "the 3090, held exclusively: loads the real EXL3 pack; run by name"]
fn the_exl3_step_decodes_the_same_on_the_planned_and_an_unaliased_arena() {
    let dir = std::env::var_os("FIFI_EXL3_CHECKPOINT").expect("FIFI_EXL3_CHECKPOINT");
    assert!(
        device_present("the_exl3_step_decodes_the_same_on_the_planned_and_an_unaliased_arena"),
        "a GPU is required"
    );
    let Pair {
        planned,
        wide,
        planned_arena_bytes,
        wide_arena_bytes,
    } = both_runs(Path::new(&dir)).expect("decode on both arenas");
    // Without aliasing in the plan the two runs are the same run and the test guards nothing.
    println!("EXL3 arena bytes: planned {planned_arena_bytes}, unaliased {wide_arena_bytes}");
    assert!(
        planned_arena_bytes < wide_arena_bytes,
        "the plan no longer aliases: {planned_arena_bytes} B planned, {wide_arena_bytes} B unaliased"
    );
    println!("EXL3 greedy ids: {:?}", planned.tokens);
    assert!(
        planned
            .tokens
            .iter()
            .all(|&id| (id as usize) < SAMPLER_VOCAB)
    );
    assert_eq!(planned.tokens, wide.tokens, "planned and unaliased ids");
    let mut worst = 0.0f64;
    for (at, (a, b)) in planned.rows.iter().zip(&wide.rows).enumerate() {
        let (a, b) = (row_values(a), row_values(b));
        assert_eq!(a.len(), HEAD_ROWS, "row {at}");
        assert!(a.iter().all(|v| v.is_finite()), "row {at} is not finite");
        let reference: Vec<f64> = b.iter().map(|&v| f64::from(v)).collect();
        let error = relative_l2(&a, &reference);
        worst = worst.max(error);
        assert!(error < ROW_BOUND, "row {at}: relative L2 {error:.3e}");
    }
    println!(
        "EXL3 max relative L2 over {} rows: {worst:.3e}",
        planned.rows.len()
    );
    if let Ok(recorded) = std::env::var("FIFI_EXL3_GREEDY_IDS") {
        let want: Vec<u32> = recorded
            .split(',')
            .map(|id| id.trim().parse().expect("FIFI_EXL3_GREEDY_IDS holds ids"))
            .collect();
        assert!(want.len() <= planned.tokens.len(), "more ids than decoded");
        assert_eq!(
            planned.tokens[..want.len()],
            want[..],
            "recorded greedy ids"
        );
    }
}

/// The EXL3 step captured in a graph decodes the same ids and the same logits bytes as the eager
/// step with the picks read on the host, and a router that stops producing valid picks makes the
/// captured token fail with the EXL3 arithmetic refusal instead of committing it: the indexed
/// launch skips the pair of an invalid id and raises the flag, and the flag rides the result pair.
#[test]
#[ignore = "the 3090, held exclusively: loads the real EXL3 pack; run by name"]
fn a_captured_exl3_step_matches_the_eager_one_and_refuses_a_poisoned_router() {
    let dir = std::env::var_os("FIFI_EXL3_CHECKPOINT").expect("FIFI_EXL3_CHECKPOINT");
    assert!(device_present("captured EXL3 step"), "a GPU is required");
    let projector = CudaProjector::new().expect("projector");
    let weights = load_exl3_dir(&projector, Path::new(&dir), Reserve::default()).expect("pack");
    let context = projector.context();
    let options = options();
    let rules = || QwenRules::new(EOS.to_vec(), SAMPLER_VOCAB, HEAD_ROWS);
    let eager = {
        let resources = StepResources::new(context, step_dims(), &inventory(), &options).unwrap();
        let mut step =
            QwenResidentStep::new(context, &resources, &weights, &options, rules().unwrap())
                .unwrap();
        decode(&mut step).expect("eager decode")
    };
    let resources = StepResources::new(context, step_dims(), &inventory(), &options).unwrap();
    let mut step =
        QwenResidentStep::new(context, &resources, &weights, &options, rules().unwrap()).unwrap();
    step.set_capture_config(CaptureConfig {
        capture: true,
        section_marks: false,
    })
    .expect("an EXL3 step accepts capture");
    let captured = decode(&mut step).expect("captured decode");
    // The first token runs eager (the warm-up), every later one the graph: a refused capture would
    // report its fallback, and then this test would compare eager with eager.
    assert_eq!(
        step.last_report().capture,
        CaptureOutcome::Captured,
        "the last token must have replayed the captured graph"
    );
    println!("EXL3 captured greedy ids: {:?}", captured.tokens);
    assert_eq!(captured.tokens, eager.tokens, "captured and eager ids");
    assert_eq!(captured.rows, eager.rows, "captured and eager logits bytes");
    // Poison the router of layer 3: every logit becomes NaN, so the router writes no valid pick.
    let table = LayerTable::new(&step_dims(), &inventory()).unwrap();
    let Some(CudaWeight::Raw { buffer, .. }) =
        weights.get(table.name(3, Role::FfnGateInp).expect("router name"))
    else {
        panic!("the router of layer 3 is a raw tensor");
    };
    let stream = context.stream().expect("stream");
    let nan: Vec<u8> = (0..buffer.bytes() / 4)
        .flat_map(|_| f32::NAN.to_le_bytes())
        .collect();
    buffer.write_at(&stream, 0, &nan).expect("poison write");
    stream.synchronize().expect("synchronize");
    match step.run_token(captured.tokens[0], true) {
        Err(EngError::ModelOutput(message)) => {
            assert!(
                message.contains("EXL3 projection flagged invalid arithmetic"),
                "{message}"
            );
        }
        other => panic!("a poisoned router must be refused with the flag error: {other:?}"),
    }
}
