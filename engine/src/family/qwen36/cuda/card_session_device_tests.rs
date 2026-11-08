//! The session-level device rig: a REAL GGUF file holding every tensor of
//! the two-layer rig spec, loaded through the production per-card loaders
//! into a [`CardStack`], wrapped in the [`CardSession`], and driven as the
//! executor would drive it - prompt feeds, logits reads, the fold, the
//! reset, and the wrapper's refusals. The block-level numerics are the
//! card token rig's proof; this rig proves the session wrapper: the step
//! params of the first token, the token-word writes of the later ones, the
//! status fold, the vocabulary-shard concat, and the turn reset.
use std::sync::Arc;

use super::card_block_device_tests::{FileEntry, write_gguf};
use super::card_session::CardSession;
use super::card_step::CardStack;
use super::dims::{KvMode, StepDims, StepOptions};
use super::layer_table::LayerTable;
use super::mixer_geometry::MixerGeometry;
use super::native_device_rig::{Rng, device_present, q4k_rows, raw_values};
use super::resources::{GlobalNames, MoeSpecs};
use super::shard::ShardTable;

use crate::EngError;
use crate::family::AttentionKind;
use crate::family::gguf_model::Inventory;
use crate::family::project::cuda::{BorrowedLoader, LoadOptions, WeightLayout, served_admission};
use crate::family::qwen36::cuda::card_load::card_loads;
use crate::family::qwen36::cuda::card_weights::load_card_weights;
use crate::family::qwen36::host::HostShape;
use crate::family::session::TokenStepper;
use crate::family::spec::{FamilySpec, QWEN3_6_35B_A3B};
use crate::weights::GgufType;
use crate::weights::{GgufLimits, GgufWeights};

/// DeltaNet then Full, the model's head pattern: the two kinds the card
/// paths cover, at a vocabulary the file can afford twice.
static TWO_LAYERS: [AttentionKind; 2] = [AttentionKind::DeltaNet, AttentionKind::Full];

fn rig_spec() -> FamilySpec {
    FamilySpec {
        layers: 2,
        vocab: 4096,
        layers_attention: &TWO_LAYERS,
        ..QWEN3_6_35B_A3B
    }
}

/// Best value and best runner-up of a logits row, for the greedy-separation
/// print.
fn top2_of(row: &[f32]) -> (f32, f32) {
    let (mut best, mut runner) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for &v in row {
        if v > best {
            runner = best;
            best = v;
        } else if v > runner {
            runner = v;
        }
    }
    (best, runner)
}

/// One q4_k stack of `experts` experts of `rows` rows of `k` values, in
/// GGUF rank-3 order `[k, rows, experts]`.
fn q4k_stack(experts: usize, rows: usize, k: usize, rng: &mut Rng) -> FileEntry {
    FileEntry {
        name: String::new(),
        ty: GgufType::Q4K,
        dims: vec![k as u64, rows as u64, experts as u64],
        payload: (0..experts).flat_map(|_| q4k_rows(rows, k, rng)).collect(),
    }
}

// One-per-process by design: the session keeps its stack, weights and states
// leaked for its whole life (the 'static owner pattern), so its two CUDA
// adapter leases never release and every later CUDA test in this process
// would fail on lease admission. Run it with --ignored, alone.
#[test]
#[ignore = "holds its two CUDA adapter leases for the process lifetime; run with --ignored, one per process"]
fn a_card_session_feeds_logits_folds_and_resets_over_a_real_gguf() {
    const NAME: &str = "a_card_session_feeds_logits_folds_and_resets_over_a_real_gguf";
    if !device_present(NAME) {
        return;
    }
    let result = run_session_rig(NAME, [0, 0]);
    result.expect("the card session rig");
}

#[test]
#[ignore = "requires two peer-enabled GPUs and holds their adapter leases for the process lifetime"]
fn prefill_without_head_matches_normal_logits_on_peer_gpus() {
    const NAME: &str = "prefill_without_head_matches_normal_logits_on_peer_gpus";
    assert!(device_present(NAME));
    let devices = [0, 1];
    let facts = crate::kernels::cuda::peer::probe_device_set(&devices).expect("probe GPUs");
    crate::kernels::cuda::peer::check_peer_facts(&facts).expect("peer access");
    crate::kernels::cuda::peer::enable_peer_access(&devices).expect("enable peer access");
    run_session_rig(NAME, devices).expect("two-GPU session parity");
}

fn run_session_rig(name: &'static str, devices: [usize; 2]) -> Result<(), EngError> {
    let spec = rig_spec();
    let inv = Inventory::for_spec(&spec)?;
    let shape = HostShape::from_spec(&spec, 64)?;
    let d = StepDims::new(&shape, &inv)?;
    let options = StepOptions {
        kv_mode: KvMode::Int8G32,
        context_tokens: 8,
        delta_slots: 1,
        forked_experts: false,
        tiered: None,
        layout: WeightLayout::Native,
    };
    let table_layers = LayerTable::new(&d, &inv)?;

    // The file: every mixer, MoE and global tensor of the spec, full-width.
    let geometry = MixerGeometry::new(&d)?;
    let moe_specs = MoeSpecs::new(&d)?;
    let mut rng = Rng::new(0x5E55);
    let mut entries: Vec<FileEntry> = Vec::new();
    for layer in 0..table_layers.len() {
        let kind = table_layers.slot(layer)?.kind;
        for spec in geometry.dense_specs(WeightLayout::Native, kind)? {
            entries.push(FileEntry {
                name: table_layers.name(layer, spec.role)?.to_owned(),
                ty: GgufType::Q4K,
                dims: vec![spec.k as u64, spec.rows as u64],
                payload: q4k_rows(spec.rows, spec.k, &mut rng),
            });
        }
        for spec in geometry.raw_specs(kind)? {
            // The conv1d is the ONE raw tensor the card loads slice by
            // CHANNEL rows, so the file carries it rank 2 - kernel values
            // wide, one row per channel - like the shipped checkpoint.
            let dims = if spec.role == crate::family::gguf_model::Role::SsmConv1d {
                vec![
                    d.shape.delta.conv_kernel as u64,
                    d.shape.delta.conv_dim() as u64,
                ]
            } else {
                vec![spec.values as u64]
            };
            entries.push(FileEntry {
                name: table_layers.name(layer, spec.role)?.to_owned(),
                ty: GgufType::F32,
                dims,
                payload: raw_values(&spec.role, spec.values)
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect(),
            });
        }
        let f32_entry = |role, values: usize| -> Result<FileEntry, EngError> {
            Ok(FileEntry {
                name: table_layers.name(layer, role)?.to_owned(),
                ty: GgufType::F32,
                dims: vec![values as u64],
                payload: raw_values(&role, values)
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect(),
            })
        };
        let (h, m) = (d.shape.hidden, &d.shape.moe);
        entries.push(f32_entry(
            crate::family::gguf_model::Role::FfnGateInp,
            moe_specs.router_values,
        )?);
        entries.push(f32_entry(
            crate::family::gguf_model::Role::FfnGateInpShexp,
            moe_specs.shexp_values,
        )?);
        for (role, experts, rows, k) in [
            (
                crate::family::gguf_model::Role::FfnGateExps,
                m.experts,
                m.inter,
                h,
            ),
            (
                crate::family::gguf_model::Role::FfnUpExps,
                m.experts,
                m.inter,
                h,
            ),
            (
                crate::family::gguf_model::Role::FfnDownExps,
                m.experts,
                h,
                m.inter,
            ),
        ] {
            let mut entry = q4k_stack(experts, rows, k, &mut rng);
            entry.name = table_layers.name(layer, role)?.to_owned();
            entries.push(entry);
        }
        for (role, rows, k) in [
            (
                crate::family::gguf_model::Role::FfnGateShexp,
                m.shared_inter,
                h,
            ),
            (
                crate::family::gguf_model::Role::FfnUpShexp,
                m.shared_inter,
                h,
            ),
            (
                crate::family::gguf_model::Role::FfnDownShexp,
                h,
                m.shared_inter,
            ),
        ] {
            entries.push(FileEntry {
                name: table_layers.name(layer, role)?.to_owned(),
                ty: GgufType::Q4K,
                dims: vec![k as u64, rows as u64],
                payload: q4k_rows(rows, k, &mut rng),
            });
        }
    }
    // The globals: the q4_k embedding (card 0), the full head (sliced per
    // card at load), and the replicated final norm.
    let names = GlobalNames::new(&inv)?;
    let (h, vocab) = (d.shape.hidden, d.shape.vocab);
    entries.push(FileEntry {
        name: names.embed.clone(),
        ty: GgufType::Q4K,
        dims: vec![h as u64, vocab as u64],
        payload: q4k_rows(vocab, h, &mut rng),
    });
    entries.push(FileEntry {
        name: names.head.clone(),
        ty: GgufType::Q4K,
        dims: vec![h as u64, vocab as u64],
        payload: q4k_rows(vocab, h, &mut rng),
    });
    entries.push(FileEntry {
        name: names.norm.clone(),
        ty: GgufType::F32,
        dims: vec![h as u64],
        payload: raw_values(&crate::family::gguf_model::Role::OutputNorm, h)
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
    });

    let dir = tempfile::tempdir().map_err(|e| EngError::Checkpoint(e.to_string()))?;
    let path = write_gguf(dir.path(), "session.gguf", &entries);
    let weights = GgufWeights::open(
        &path,
        GgufLimits {
            max_header_bytes: 1 << 22,
            max_tensor_bytes: 1 << 30,
        },
        None,
    )?;

    // The stack and the per-card loads, through the production loaders on
    // the cards' own contexts.
    let admission = served_admission();
    let shard = ShardTable::new(&d.shape, 2, 32)?;
    let loads = card_loads(&shard, &inv, GgufType::Q4K)?;
    let (mut stack, states) = CardStack::new(&d, &options, &devices, &admission)?;
    stack.enable_broadcast_sync_for_test();
    let load_options = LoadOptions {
        layout: WeightLayout::Native,
        ..LoadOptions::default()
    };
    let mut owned: Vec<&'static mut crate::family::project::cuda::CudaWeights> = Vec::new();
    for (card, load) in loads.iter().enumerate() {
        let loader = BorrowedLoader::new(&stack.contexts()[card])?;
        let weights = load_card_weights(&loader, &weights, load, card, load_options)?;
        owned.push(Box::leak(Box::new(weights)));
    }
    // The leaked slice of shared refs: one per engine, like the start's
    // leaks. The mut handles convert by consuming them.
    let loaded: &'static [&'static crate::family::project::cuda::CudaWeights] = Box::leak(
        owned
            .into_iter()
            .map(|weights| &*weights)
            .collect::<Vec<_>>()
            .into_boxed_slice(),
    );
    let stack: &'static CardStack = Box::leak(Box::new(stack));
    let table_static: &'static LayerTable = Box::leak(Box::new(table_layers));
    let eos: Arc<[u32]> = Arc::from(vec![7u32]);
    let mut session = CardSession::new(
        stack,
        states,
        loaded,
        table_static,
        &inv,
        &d,
        &options,
        eos,
        vocab,
        crate::family::session::StoreLimits::default(),
    )?;
    session.set_skip_prefill_head_for_test(false);
    let limit = session.context_limit();
    if limit < 4 {
        return Err(EngError::Checkpoint(format!(
            "{name}: the fixture context limit {limit} is too small for the rig"
        )));
    }

    // The prompt feeds: no read-back, the fed book advances.
    session.feed(0)?;
    session.feed(1)?;
    let row = session.logits(2)?;
    assert_eq!(session.fed(), 3);
    assert_eq!(row.len(), vocab, "{name}: the head shards rebuild the row");
    assert!(row.iter().all(|v| v.is_finite()), "{name}: finite logits");
    let (best, runner) = top2_of(&row);
    println!(
        "{name}: greedy gap after 3 tokens: {:.3} (runner {runner:.3}, best {best:.3})",
        best - runner
    );

    // The out-of-vocabulary token refuses by name and changes nothing.
    let refused = session
        .logits(vocab as u32)
        .expect_err("out-of-vocab token");
    assert!(
        matches!(refused, EngError::BadRequest(_)),
        "{name}: the wrapper refuses the id, got {refused:?}"
    );
    assert_eq!(session.fed(), 3);

    // Past the limit the session refuses: feed to the limit then one more.
    for token in 3..limit as u32 {
        session.feed(token)?;
    }
    let refused = session
        .feed(limit as u32)
        .expect_err("past the context limit");
    assert!(
        matches!(refused, EngError::BadRequest(_)),
        "{name}: the limit refusal, got {refused:?}"
    );

    // The turn reset: same tokens again, the same logits row bit for bit.
    TokenStepper::reset(&mut session)?;
    assert_eq!(session.fed(), 0);
    session.feed(0)?;
    session.feed(1)?;
    let row_after = session.logits(2)?;
    assert_eq!(
        row, row_after,
        "{name}: the reset run reproduces the logits row exactly"
    );

    // Record a later decode row after another prompt feed. The optimized
    // path must preserve state both before and after a logits-producing step.
    session.feed(3)?;
    let next_row = session.logits(4)?;
    TokenStepper::reset(&mut session)?;
    session.set_skip_prefill_head_for_test(true);
    session.feed(0)?;
    session.feed(1)?;
    let optimized_row = session.logits(2)?;
    assert_eq!(row, optimized_row, "{name}: first decoded logits");
    session.feed(3)?;
    let optimized_next = session.logits(4)?;
    assert_eq!(next_row, optimized_next, "{name}: later decoded logits");

    // A corrupt head makes a logits-producing step fail. A prompt-only feed
    // must still succeed with that head, proving that it was not launched.
    let head = match loaded[0].get(&names.head) {
        Some(crate::family::project::cuda::CudaWeight::Native { weight, .. }) => weight.buffer(),
        _ => {
            return Err(EngError::Checkpoint(
                "test fixture has no native head".into(),
            ));
        }
    };
    let stream = &stack.streams()[0];
    let mut saved = vec![0u8; head.bytes()];
    head.read_at(stream, 0, &mut saved)?;
    head.write_at(stream, 0, &vec![0xff; saved.len()])?;
    TokenStepper::reset(&mut session)?;
    let feed_result = session.feed(0);
    let logits_result = if feed_result.is_ok() {
        Some(session.logits(1))
    } else {
        None
    };
    head.write_at(stream, 0, &saved)?;
    feed_result?;
    assert!(
        matches!(logits_result, Some(Err(EngError::Backend { .. }))),
        "{name}: the decode must still reject a non-finite head"
    );
    Ok(())
}
