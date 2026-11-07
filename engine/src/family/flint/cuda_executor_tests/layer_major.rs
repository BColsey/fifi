//! Layer-major prefill at the executor: the CPU-checkable plumbing (admission,
//! piece hint, mode-bound prefix fingerprint) and the device gates that need
//! a GPU. The device tests are `#[ignore]`; they run with `FLINT_EXL3_PACK`
//! set, `FIFI_FLINT_LATE_HEALTH` pinned to 0 or 1, and the expert cache
//! enabled as in serving.

use super::*;
use crate::family::flint::kv_plan::LAYERS;
use serde_json::json;

const GIB: u64 = 1 << 30;
const MIB: u64 = 1 << 20;

fn options(context: usize) -> FlintOptions {
    options_for(context, "f16")
}

fn options_for(context: usize, profile: &str) -> FlintOptions {
    serde_json::from_value(json!({
        "context_limit": context, "kv_profile": profile, "prefill_chunk_size": 8
    }))
    .unwrap()
}

#[test]
fn the_piece_is_as_tall_as_the_router_admits_and_never_taller_than_the_context() {
    assert_eq!(layer_major_piece_rows(262_144), 4096);
    assert_eq!(layer_major_piece_rows(65_536), 4096);
    assert_eq!(layer_major_piece_rows(4096), 4096);
    assert_eq!(layer_major_piece_rows(576), 576);
    assert_eq!(layer_major_piece_rows(1), 1);
}

#[test]
fn staging_charges_one_piece_not_the_whole_context() {
    let extra = State::layer_major_extra_bytes(4096, false).unwrap();
    // A piece of 4,096 rows: 384 MiB of faces plus the planes and tile
    // scratch, whatever the context — a longer prompt is more pieces, not a
    // taller staging.
    assert_eq!(
        layer_major_staging_charge(4096, false).unwrap(),
        384 * MIB + extra
    );
    assert_eq!(
        layer_major_staging_charge(576, false).unwrap(),
        State::layer_major_staging_bytes(576).unwrap()
            + State::layer_major_extra_bytes(576, false).unwrap()
    );
    assert!(extra < GIB, "extra staging {extra}");
}

#[test]
fn every_context_the_loader_can_build_is_admitted_armed() {
    // The refusal itself is pinned by the admission test over explicit terms.
    for context in [4096, 32_768, 65_536, 262_144] {
        let plan = KvPlan::new(context, 8, true).unwrap();
        let staging = layer_major_staging_charge(layer_major_piece_rows(context), false).unwrap();
        layer_major_admission(
            "FIFI_EXL3_LAYER_MAJOR=1",
            plan.bytes(plan.storage().row_bytes()).unwrap() as u64,
            work_reserve(&plan, 8, 0).unwrap(),
            staging,
            512 << 20,
            LAYER_MAJOR_CEILING_BYTES,
        )
        .unwrap_or_else(|error| panic!("context {context} refused: {error:?}"));
    }
}

#[test]
fn the_worker_hint_is_the_piece_only_when_armed() {
    assert_eq!(prefill_hint(None, 8), 8);
    assert_eq!(prefill_hint(None, 1), 1);
    assert_eq!(prefill_hint(Some(4096), 8), 4096);
    assert_eq!(prefill_hint(Some(576), 8), 576);
}

fn env(layer_major: LayerMajor) -> CudaPrefixEnv {
    CudaPrefixEnv {
        alloc_cache: true,
        late_health: false,
        packed_pointwise: true,
        exl3_legacy: false,
        expert_cache: true,
        layer_major,
        tall_hilo: false,
    }
}

fn path(layer_major: LayerMajor) -> CudaPrefixPath {
    CudaPrefixPath::from_values(Path::new("/model"), 4096, 8, "f16", env(layer_major))
}

/// A model with no native state: only the position count the save checks.
#[derive(Default)]
struct HostStep {
    fed: usize,
}

impl TokenStepper for HostStep {
    const OPERATION: &'static str = "layer_major_test";

    fn feed(&mut self, _token: u32) -> Result<()> {
        self.fed += 1;
        Ok(())
    }

    fn logits(&mut self, _token: u32) -> Result<Vec<f32>> {
        Ok(vec![0.0])
    }

    fn fed(&self) -> usize {
        self.fed
    }

    fn reset(&mut self) -> Result<()> {
        self.fed = 0;
        Ok(())
    }
}

fn snapshot() -> CudaPrefixSnapshot {
    CudaPrefixSnapshot {
        layers: Vec::new(),
        health: None,
        committed: 9,
    }
}

#[test]
fn a_prefix_saved_in_one_mode_never_restores_in_another() {
    let prompt: Vec<u32> = (0..10).collect();
    let longer: Vec<u32> = (0..14).collect();
    let modes = [LayerMajor::Off, LayerMajor::Tiles, LayerMajor::Jobs];
    for saved_armed in modes {
        for restoring_armed in modes {
            let mut store = PrefixStore::<CudaPrefixSnapshot, CudaPrefixPath>::with_limits(
                StoreLimits::default(),
            );
            let mut writer = DecodeSession::default();
            let mut step = HostStep::default();
            writer.prefill(&mut step, &prompt, 4096).unwrap();
            assert!(
                writer
                    .save_prefix(
                        &mut store,
                        &prompt,
                        path(saved_armed),
                        step.fed(),
                        SaveKind::Tip,
                        || Ok(snapshot()),
                    )
                    .unwrap()
            );
            let mut reader = DecodeSession::default();
            let mut opened = None;
            let restored = reader
                .restore_prefix(&mut store, &longer, path(restoring_armed), |entry| {
                    opened = Some(entry.committed);
                    Ok(())
                })
                .unwrap();
            if restoring_armed == saved_armed {
                assert_eq!(restored, prompt.len(), "the same mode restores");
                assert_eq!(opened, Some(9));
            } else {
                assert_eq!(
                    restored, 0,
                    "saved {saved_armed:?}, restoring {restoring_armed:?}"
                );
                assert_eq!(opened, None, "the snapshot is never opened across modes");
            }
        }
    }
}

fn pack_root() -> PathBuf {
    PathBuf::from(std::env::var_os("FLINT_EXL3_PACK").expect("opt-in real EXL3 pack"))
}

/// An armed executor, its armed initial session, and a second session whose
/// state is not armed: it can only be fed through [`TokenMajor`].
fn armed_executor(context: usize) -> (FlintExecutor, FlintSession, FlintSession) {
    armed_executor_in(context, LayerMajor::Tiles)
}

/// [`armed_executor`] for either layer-major pass.
fn armed_executor_in(
    context: usize,
    mode: LayerMajor,
) -> (FlintExecutor, FlintSession, FlintSession) {
    armed_executor_with(context, mode, "f16")
}

/// [`armed_executor_in`] on the f16 or the q8 KV profile.
fn armed_executor_with(
    context: usize,
    mode: LayerMajor,
    profile: &str,
) -> (FlintExecutor, FlintSession, FlintSession) {
    let piece = match mode {
        LayerMajor::Jobs => 1024.min(context),
        _ => layer_major_piece_rows(context),
    };
    let (executor, armed) = FlintExecutor::open_with(
        &pack_root(),
        &options_for(context, profile),
        vec![2],
        VOCAB,
        [0u8; 32],
        || Ok(Some(mode)),
    )
    .unwrap();
    assert_eq!(executor.layer_major, Some(piece));
    assert_eq!(executor.prefill_piece(), Some(piece));
    // Open charged the executor and its armed session; the comparison session
    // is extra KV and scratch, so admit it before allocating, as the other
    // two-session fixtures do.
    let options = options_for(context, profile);
    let plan = options.plan().unwrap();
    let cache_bytes = CudaContext::alloc_cache_reserve(executor.ctx.alloc_cache_enabled());
    plan::admit(
        &plan::Demand {
            tensors: 0,
            kv_cache: plan.bytes(plan.storage().row_bytes()).unwrap() as u64,
            work: work_reserve(&plan, options.prefill_chunk_size, cache_bytes).unwrap(),
            headroom: 512 << 20,
        },
        executor.ctx.memory_info().unwrap().free as u64,
    )
    .unwrap();
    let reference = FlintSession {
        state: State::new(&executor.ctx, plan).unwrap(),
        core: DecodeSession::default(),
    };
    (executor, armed, reference)
}

// Which walk fed a piece is visible in the order its KV appends commit. The
// token-major walk commits all 49 layers once per chunk, chunk after chunk;
// the layer-major pass commits every block of a layer before the next layer.
// A test that asserts the second shape cannot pass on the first.

fn chunk_ends(start: usize, rows: usize, chunk: usize) -> Vec<usize> {
    (1..=rows.div_ceil(chunk))
        .map(|index| start + (index * chunk).min(rows))
        .collect()
}

fn token_major_ends(start: usize, rows: usize, chunk: usize) -> Vec<usize> {
    chunk_ends(start, rows, chunk)
        .into_iter()
        .flat_map(|end| std::iter::repeat_n(end, LAYERS))
        .collect()
}

fn layer_major_ends(start: usize, rows: usize, chunk: usize, layers: usize) -> Vec<usize> {
    let ends = chunk_ends(start, rows, chunk);
    (0..layers).flat_map(|_| ends.clone()).collect()
}

fn commits_of<T>(call: impl FnOnce() -> T) -> (T, Vec<usize>) {
    let before = test_commit_ends().len();
    let result = call();
    (result, test_commit_ends()[before..].to_vec())
}

#[test]
fn commit_traces_tell_the_two_walks_apart() {
    assert_eq!(
        layer_major_ends(0, 9, 8, LAYERS),
        [8, 9].repeat(LAYERS),
        "layer-major: both blocks of a layer, then the next layer"
    );
    assert_eq!(
        token_major_ends(0, 9, 8),
        [vec![8; LAYERS], vec![9; LAYERS]].concat(),
        "token-major: 49 commits at each boundary"
    );
    // The token-major model is the shape the real-pack B8 gate pins.
    assert_eq!(
        token_major_ends(0, 21, 8),
        [8, 16, 21]
            .into_iter()
            .flat_map(|end| std::iter::repeat_n(end, 49))
            .collect::<Vec<_>>()
    );
    // Above one block the traces always differ, so asserting the layer-major
    // one rejects a pass that fell back to the token-major walk; at or below
    // one block they coincide, which is why those pieces assert the
    // token-major shape and the tails assert the other.
    for rows in [1, 7, 8] {
        assert_eq!(
            layer_major_ends(0, rows, 8, LAYERS),
            token_major_ends(0, rows, 8)
        );
    }
    for rows in [9, 63, 64, 65, 700, 4096] {
        assert_ne!(
            layer_major_ends(0, rows, 8, LAYERS),
            token_major_ends(0, rows, 8),
            "rows {rows}"
        );
    }
    // An abort at layer 7 leaves eight layers of commits layer-major, none
    // token-major (its commits wait for the whole pass).
    assert_eq!(layer_major_ends(0, 64, 8, 8).len(), 8 * 8);
}

/// The token-major walk on an armed executor: eight-row chunks, whatever the
/// piece size, so the same executor produces both sides of a comparison.
struct TokenMajor<'a>(Step<'a>);

impl TokenStepper for TokenMajor<'_> {
    const OPERATION: &'static str = "flint_cuda";

    fn feed(&mut self, token: u32) -> Result<()> {
        self.0.feed(token)
    }

    fn feed_rows(&mut self, tokens: &[u32]) -> Result<()> {
        let decoder = self.0.executor.decoder()?;
        for chunk in tokens.chunks(self.0.executor.prefill_chunk_size) {
            decoder.feed_rows(self.0.state, chunk)?;
        }
        Ok(())
    }

    fn logits(&mut self, token: u32) -> Result<Vec<f32>> {
        self.0.logits(token)
    }

    fn fed(&self) -> usize {
        self.0.fed()
    }

    fn reset(&mut self) -> Result<()> {
        self.0.reset()
    }
}

fn prefill_token_major(executor: &FlintExecutor, session: &mut FlintSession, tokens: &[u32]) {
    let mut step = TokenMajor(Step {
        executor,
        state: &mut session.state,
    });
    session
        .core
        .prefill(&mut step, tokens, executor.context_limit)
        .unwrap();
}

fn plane_values(executor: &FlintExecutor, plane: &DeviceBuffer, rows: Range<usize>) -> Vec<f32> {
    canonical_plane(&executor.stream, plane, rows)
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| half::f16::from_bits(u16::from_ne_bytes(*pair)).to_f32())
        .collect()
}

fn difference(reference: &[f32], candidate: &[f32]) -> (f32, f32) {
    assert_eq!(reference.len(), candidate.len());
    let scale = reference.iter().map(|v| v.abs()).sum::<f32>() / reference.len() as f32;
    let (mut worst, mut total) = (0f32, 0f32);
    for (a, b) in reference.iter().zip(candidate) {
        assert!(a.is_finite() && b.is_finite());
        worst = worst.max((a - b).abs());
        total += (a - b).abs();
    }
    (
        worst,
        total / reference.len() as f32 / scale.max(f32::MIN_POSITIVE),
    )
}

/// Compares every layer's committed window of two sessions and prints the
/// figures; returns the worst mean relative difference. Acceptance of the
/// numbers belongs to the teacher-forced gate, not to this structural check.
fn compare_kv(executor: &FlintExecutor, reference: &FlintSession, candidate: &FlintSession) -> f32 {
    let fed = reference.state.fed();
    assert_eq!(candidate.state.fed(), fed);
    let mut worst_mean = 0f32;
    for (layer, (left, right)) in reference
        .state
        .test_kv()
        .iter()
        .zip(candidate.state.test_kv())
        .enumerate()
    {
        assert_eq!(
            left.cursor.committed_prefix(),
            fed,
            "reference layer {layer}"
        );
        assert_eq!(
            right.cursor.committed_prefix(),
            fed,
            "candidate layer {layer}"
        );
        let start = if FULL_LAYERS.contains(&(layer as u64)) {
            0
        } else {
            fed.saturating_sub(513)
        };
        for (name, a, b) in [
            ("K", &left.keys, &right.keys),
            ("V", &left.values, &right.values),
        ] {
            let (worst, mean) = difference(
                &plane_values(executor, a, start..fed),
                &plane_values(executor, b, start..fed),
            );
            worst_mean = worst_mean.max(mean);
            if layer % 12 == 0 {
                println!("layer {layer} {name}: max abs {worst:e}, mean rel {mean:e}");
            }
        }
    }
    worst_mean
}

fn next_logits(executor: &FlintExecutor, session: &mut FlintSession) -> Vec<f32> {
    let pending = session.core.checked_pending().unwrap();
    executor.qual_logits(session, pending).unwrap()
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

#[test]
#[ignore = "requires admitted GPU and full EXL3 pack; `off` opens the token-major walk"]
fn the_off_knob_opens_the_token_major_walk_with_no_staging() {
    let (executor, session) =
        FlintExecutor::open_token_major(&pack_root(), &options(CONTEXT), vec![2], VOCAB, [0u8; 32])
            .unwrap();
    assert_eq!(executor.layer_major, None);
    assert_eq!(executor.prefill_piece(), Some(8));
    assert!(session.state.test_layer_major_rows().is_none());
}

#[test]
#[ignore = "requires admitted GPU and full EXL3 pack; the environment decides, run with FIFI_EXL3_LAYER_MAJOR unset"]
fn the_unset_knob_opens_the_jobs_pass_on_f16_and_the_token_major_walk_on_q8() {
    // The environment decides here, exactly as serving does.
    assert!(std::env::var_os("FIFI_EXL3_LAYER_MAJOR").is_none());
    let piece = 1024.min(CONTEXT);
    let (executor, session) =
        FlintExecutor::open(&pack_root(), &options(CONTEXT), vec![2], VOCAB, [0u8; 32]).unwrap();
    assert_eq!(executor.layer_major, Some(piece));
    assert_eq!(session.state.test_layer_major_rows(), Some(piece));
    drop((executor, session));
    // The jobs pass is measured and gated on f16 only: q8 keeps the walk it always had.
    let (executor, session) = FlintExecutor::open(
        &pack_root(),
        &options_for(CONTEXT, "q8"),
        vec![2],
        VOCAB,
        [0u8; 32],
    )
    .unwrap();
    assert_eq!(executor.layer_major, None);
    assert!(session.state.test_layer_major_rows().is_none());
}

#[test]
#[ignore = "requires admitted GPU and full EXL3 pack; a piece of one narrow block is the token-major walk"]
fn armed_pieces_of_one_block_are_bitwise_the_token_major_walk() {
    let (mut executor, mut armed, mut reference) = armed_executor(CONTEXT);
    for rows in [1usize, 7, 8] {
        let prompt = fixture_tokens(rows + 1);
        executor.reset(&mut armed).unwrap();
        executor.reset(&mut reference).unwrap();
        let (result, trace) = commits_of(|| executor.prefill(&mut armed, &prompt));
        result.unwrap();
        assert_eq!(trace, token_major_ends(0, rows, 8), "rows {rows}");
        prefill_token_major(&executor, &mut reference, &prompt);
        assert_same_kv(&executor, &reference, &armed);
        assert_eq!(
            bits(&next_logits(&executor, &mut reference)),
            bits(&next_logits(&executor, &mut armed)),
            "rows {rows}"
        );
    }
}

#[test]
#[ignore = "requires admitted GPU and full EXL3 pack; tail heights of the regrouped walk against the token-major walk"]
fn armed_tails_track_the_token_major_walk_and_repeat_bitwise() {
    let (mut executor, mut armed, mut reference) = armed_executor(CONTEXT);
    for rows in [9usize, 63, 64, 65] {
        let prompt = fixture_tokens(rows + 1);
        executor.reset(&mut armed).unwrap();
        executor.reset(&mut reference).unwrap();
        let (result, trace) = commits_of(|| executor.prefill(&mut armed, &prompt));
        result.unwrap();
        assert_eq!(
            trace,
            layer_major_ends(0, rows, 8, LAYERS),
            "rows {rows}: the armed piece must commit layer by layer"
        );
        let ((), reference_trace) =
            commits_of(|| prefill_token_major(&executor, &mut reference, &prompt));
        assert_eq!(reference_trace, token_major_ends(0, rows, 8), "rows {rows}");
        assert_eq!(armed.state.fed(), rows);
        assert_eq!(armed.core.pending(), Some(prompt[rows]));
        let worst = compare_kv(&executor, &reference, &armed);
        let (left, right) = (
            next_logits(&executor, &mut reference),
            next_logits(&executor, &mut armed),
        );
        let (worst_logit, mean_logit) = difference(&left, &right);
        println!(
            "rows {rows}: KV worst mean rel {worst:e}; logits max abs {worst_logit:e}, mean rel {mean_logit:e}; argmax {} / {}",
            argmax(&left),
            argmax(&right)
        );
        assert_eq!(argmax(&left), argmax(&right), "rows {rows}");
        assert!(
            worst < 5e-2,
            "rows {rows}: KV drifted far from the walk it regroups"
        );
        executor.reset(&mut armed).unwrap();
        executor.prefill(&mut armed, &prompt).unwrap();
        let again = next_logits(&executor, &mut armed);
        executor.reset(&mut armed).unwrap();
        executor.prefill(&mut armed, &prompt).unwrap();
        assert_eq!(
            bits(&again),
            bits(&next_logits(&executor, &mut armed)),
            "rows {rows}"
        );
    }
}

#[test]
#[ignore = "requires admitted GPU and full EXL3 pack; ring wrap inside one armed piece, then token-major continuation"]
fn armed_piece_across_the_ring_wrap_commits_every_layer_and_continues_token_major() {
    let context = 1280;
    let (mut executor, mut armed, mut reference) = armed_executor(context);
    // 700 rows wrap the 520-row sliding rings inside a single piece.
    let prompt = fixture_tokens(701);
    let (result, trace) = commits_of(|| executor.prefill(&mut armed, &prompt));
    result.unwrap();
    assert_eq!(
        trace,
        layer_major_ends(0, 700, 8, LAYERS),
        "the 700-row piece must commit layer by layer"
    );
    prefill_token_major(&executor, &mut reference, &prompt);
    assert_eq!(armed.state.fed(), 700);
    assert!(
        armed
            .state
            .test_kv()
            .iter()
            .all(|cache| cache.cursor.committed_prefix() == 700 && !cache.cursor.test_poisoned())
    );
    let worst = compare_kv(&executor, &reference, &armed);
    assert!(worst < 5e-2, "ring KV drifted: {worst:e}");
    // The next piece is short, so it runs token-major from the armed position.
    let tail = fixture_tokens(704);
    let (result, trace) = commits_of(|| executor.prefill(&mut armed, &tail[701..]));
    result.unwrap();
    assert_eq!(trace, token_major_ends(700, 3, 8));
    assert_eq!(armed.state.fed(), 703);
}

#[test]
#[ignore = "requires admitted GPU and full EXL3 pack; poison, quiescent reset and bitwise replay of an armed piece"]
fn armed_failure_poisons_until_reset_and_replays_bitwise() {
    use crate::family::flint::cuda_ffn::device::route_trace::inject;
    let (mut executor, mut armed, _reference) = armed_executor(CONTEXT);
    let prompt = fixture_tokens(65);
    executor.prefill(&mut armed, &prompt).unwrap();
    let pristine = bits(&next_logits(&executor, &mut armed));
    executor.reset(&mut armed).unwrap();
    for (name, set) in [
        (
            "after the routed FFN",
            inject::set_after_ffn_abort as fn(Option<usize>),
        ),
        ("after the residual", inject::set_after_residual_abort),
    ] {
        set(Some(7));
        let (refused, trace) = commits_of(|| executor.prefill(&mut armed, &prompt));
        set(None);
        // Layers 0..=7 appended and committed their 64 rows before the hook
        // fired; the token-major walk would have aborted its first chunk with
        // nothing committed.
        assert_eq!(trace, layer_major_ends(0, 64, 8, 8), "{name}");
        assert_eq!(inject::take_aborts_fired(), 1, "{name}: the hook ran once");
        assert!(
            matches!(refused, Err(EngError::Backend { .. })),
            "{name}: {refused:?}"
        );
        assert!(armed.state.test_poisoned(), "{name}");
        executor
            .ctx
            .ensure_usable()
            .expect("a session failure is not a backend terminal");
        assert!(matches!(
            executor.prefill(&mut armed, &prompt),
            Err(EngError::BadRequest(_))
        ));
        executor.reset(&mut armed).unwrap();
        assert!(!armed.state.test_poisoned());
        assert!(
            armed
                .state
                .test_kv()
                .iter()
                .all(|cache| cache.cursor.committed_prefix() == 0 && !cache.cursor.test_poisoned())
        );
        executor.prefill(&mut armed, &prompt).unwrap();
        assert_eq!(
            bits(&next_logits(&executor, &mut armed)),
            pristine,
            "{name}: replay after reset"
        );
        executor.reset(&mut armed).unwrap();
    }
}

#[test]
#[ignore = "requires admitted GPU and full EXL3 pack; worker-sized pieces: a full 4096-row piece, then a short tail piece"]
fn armed_worker_pieces_split_at_the_hint_and_keep_the_pending_token() {
    let context = 8192;
    let (mut executor, mut armed, mut reference) = armed_executor(context);
    let piece = executor.prefill_piece().unwrap();
    assert_eq!(piece, 4096);
    // One token primes the pending slot, so the next chunk of 4096 tokens
    // feeds the pending token plus 4095 new ones: exactly 4096 rows.
    let prompt = fixture_tokens(1 + piece + 5);
    let (first, rest) = prompt.split_at(1);
    let (body, tail) = rest.split_at(piece);
    let (result, trace) = commits_of(|| executor.prefill(&mut armed, first));
    result.unwrap();
    assert!(trace.is_empty(), "a lone token feeds nothing");
    assert_eq!(
        (armed.state.fed(), armed.core.pending()),
        (0, Some(first[0]))
    );
    let (result, trace) = commits_of(|| executor.prefill(&mut armed, body));
    result.unwrap();
    assert_eq!(
        trace,
        layer_major_ends(0, piece, 8, LAYERS),
        "the pass must hold exactly {piece} rows, committed layer by layer"
    );
    assert_eq!(armed.state.fed(), piece);
    assert_eq!(armed.core.pending(), Some(body[piece - 1]));
    // The short tail piece is one pending row plus four: token-major.
    let (result, trace) = commits_of(|| executor.prefill(&mut armed, tail));
    result.unwrap();
    assert_eq!(trace, token_major_ends(piece, 5, 8));
    assert_eq!(armed.state.fed(), piece + 5);
    assert_eq!(executor.history(&armed), prompt.as_slice());
    prefill_token_major(&executor, &mut reference, &prompt);
    let worst = compare_kv(&executor, &reference, &armed);
    let (left, right) = (
        next_logits(&executor, &mut reference),
        next_logits(&executor, &mut armed),
    );
    println!(
        "{} rows: KV worst mean rel {worst:e}; argmax {} / {}",
        piece + 5,
        argmax(&left),
        argmax(&right)
    );
    assert_eq!(argmax(&left), argmax(&right));
    assert!(worst < 5e-2);
}

/// Real text for the piece-boundary gate: `FLINT_WINDOW_TOKENS` names a JSON array of token ids.
fn real_tokens(length: usize) -> Vec<u32> {
    let path = std::env::var_os("FLINT_WINDOW_TOKENS").expect("real token ids");
    let all: Vec<u32> = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    assert!(all.len() >= length);
    all[..length].to_vec()
}

#[test]
#[ignore = "requires admitted GPU, full EXL3 pack and FLINT_WINDOW_TOKENS; jobs pieces across boundaries through a small bank"]
fn jobs_pieces_across_boundaries_track_the_token_major_walk_through_a_bank() {
    jobs_pieces_through_a_bank("f16");
}

#[test]
#[ignore = "requires admitted GPU, full EXL3 pack and FLINT_WINDOW_TOKENS; q8 jobs pieces across boundaries through a small bank"]
fn q8_jobs_pieces_across_boundaries_track_the_q8_token_major_walk_through_a_bank() {
    jobs_pieces_through_a_bank("q8");
}

/// The piece-boundary gate on one KV profile. The K/V comparison reads f16 planes, so the q8
/// profile is judged on its final logits and its bank copies.
fn jobs_pieces_through_a_bank(profile: &str) {
    // 192 of 256 experts stay on the device: a 1024-row piece needs more than that in every
    // layer, so each piece runs a round of copies behind its resident round.
    // The runner pins the bank, as it pins the late-health arm: the test cannot set the
    // process environment itself.
    assert_eq!(
        std::env::var("FIFI_FLINT_CUDA_EXPERT_SLOTS").as_deref(),
        Ok("192"),
        "the GPU runner must pin FIFI_FLINT_CUDA_EXPERT_SLOTS=192"
    );
    let context = 3072;
    let (mut executor, mut jobs, mut reference) =
        armed_executor_with(context, LayerMajor::Jobs, profile);
    assert!(executor.model.banked_experts() == Some(192));
    // Two full pieces at fed 0 and fed 1024, a 52-row tail on the jobs pass, a six-row tail on
    // the token-major walk. Each call feeds the pending row plus all but the last new token.
    let tokens = real_tokens(1 + 1024 + 1024 + 52 + 6 + 1);
    let cuts = [1025, 2049, 2101, 2107, 2108];
    let mut done = 0;
    for (index, &cut) in cuts.iter().enumerate() {
        let part = &tokens[done..cut];
        let rows = if index == 0 { cut - 1 } else { cut - done };
        let (result, trace) = commits_of(|| executor.prefill(&mut jobs, part));
        result.unwrap();
        prefill_token_major(&executor, &mut reference, part);
        assert_eq!(jobs.state.fed(), reference.state.fed(), "cut {cut}");
        println!(
            "cut {cut}: {rows} rows fed, {} commits, fed {}",
            trace.len(),
            jobs.state.fed()
        );
        if profile == "f16" {
            let worst = compare_kv(&executor, &reference, &jobs);
            println!("cut {cut}: KV worst mean rel {worst:e}");
            // Printed for the record; only a cache unrelated to the walk's fails here.
            assert!(worst < 0.6, "cut {cut}: KV drifted: {worst:e}");
        }
        done = cut;
    }
    let (left, right) = (
        next_logits(&executor, &mut reference),
        next_logits(&executor, &mut jobs),
    );
    let (worst_logit, mean_logit) = difference(&left, &right);
    println!(
        "final logits: max abs {worst_logit:e}, mean rel {mean_logit:e}, argmax {} / {}",
        argmax(&left),
        argmax(&right)
    );
    assert_eq!(argmax(&left), argmax(&right));
    // The bank really copied: a piece larger than the bank cannot be served from residents alone.
    let bank = executor.model.bank_stats().expect("a banked model");
    println!("bank copies {} ({} bytes)", bank.misses, bank.bytes);
    assert!(bank.misses > 0);
}
