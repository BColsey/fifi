//! The first executing distributed block: one full-attention mixer block
//! on `CardStack`, two logical contexts, against the single-device
//! reference over the same Q4_K sources, the same residual row and the
//! same position - through the production launchers (`Mixers::
//! run_local_op` and the `CardStack` ring entries), never a private
//! re-wiring.
//!
//! Stage evidence (the acceptance table of the assessment):
//! * the K/V projections, prep+RoPE, gathers and append are proven
//!   TRANSITIVELY and EXACTLY by the KV cache: the card cache planes
//!   (codes and scales) equal the reference planes byte for byte, so the
//!   card's projected slices, its absolute-position RoPE and the gathered
//!   full rows produced the same appended cache;
//! * the gathered rows are also bit-identical ACROSS cards;
//! * the attention core (per-head decode over the replicated cache) is
//!   compared against the reference's head slice - same inputs, same
//!   per-head arithmetic, expected exact;
//! * the out projection and `MixerSum` split one matvec into per-card
//!   partial sums, so `MixOut` and the residual epilogue are NUMERICAL
//!   comparisons (max difference reported), not bit identities.
//!
//! The reference shares card 0's context (the process admits only two).
use super::card_block::run_card_block;
use super::card_step::CardStack;
use super::common_tests::{dims as step_dims, inventory};
use super::dims::{KvMode, StepOptions};
use super::layer_table::LayerTable;
use super::mixer::{BlockIo, CardKindOps, MixerParts, MixerViews, Mixers};
use super::mixer_geometry::MixerGeometry;
use super::mixer_ops::CardOp;
use super::mixer_weights::{DenseWeight, LayerWeights, MixerWeights};
use super::native_device_rig::{Rng, device_present, q4k_rows, raw_values};
use super::shard::ShardTable;
use super::state::ResidentState;
use super::step_plan::{KernelScratch, Slot, StepPlan};
use crate::EngError;
use crate::family::project::cuda::WeightLayout;
use crate::family::project::cuda::served_admission;
use crate::family::qwen36::cuda::state_layout::Plane;
use crate::family::{AttentionKind, gguf_model::Role};
use crate::kernels::cuda::plan::{assemble_slice, plan_slice};
use crate::kernels::cuda::resident::step_params::StepParams;
use crate::kernels::cuda::{CudaContext, DeviceBuffer, NativeWeight, QuantTensor};
use crate::weights::GgufType;

const F32: usize = size_of::<f32>();
const Q4K_BLOCK: usize = 144;
const Q4K_VALUES: usize = 256;
/// Generous envelope for the split-sum stages: the partial matvecs differ
/// from the fused one by reduction order only, orders of magnitude below
/// the row amplitudes.
const SPLIT_SUM_RELATIVE: f32 = 1e-3;

fn options() -> StepOptions {
    StepOptions {
        kv_mode: KvMode::Int8G32,
        context_tokens: 8,
        delta_slots: 1,
        forked_experts: false,
        tiered: None,
        layout: WeightLayout::Native,
    }
}

fn upload_native(
    ctx: &CudaContext,
    source: &[u8],
    rows: usize,
    k: usize,
) -> Result<NativeWeight, EngError> {
    let tensor = QuantTensor {
        name: "rig",
        ty: GgufType::Q4K,
        n_rows: rows,
        k,
        source_len: source.len(),
    };
    let mut reader = |at: usize, buf: &mut [u8]| {
        buf.copy_from_slice(&source[at..at + buf.len()]);
        Ok(())
    };
    ctx.upload_native(tensor, 1 << 16, &mut reader)
}

/// The row blocks of `rows` of a Q4_K source of `k` columns.
fn row_slice(source: &[u8], k: usize, rows: std::ops::Range<usize>) -> Vec<u8> {
    let per_row = k / Q4K_VALUES * Q4K_BLOCK;
    source[rows.start * per_row..rows.end * per_row].to_vec()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn max_abs(values: &[f32]) -> f32 {
    values.iter().fold(0.0f32, |m, v| m.max(v.abs()))
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// The whole comparison, `?` inside; asserts live in the caller.
struct Evidence {
    name: &'static str,
    mix_out_ref: Vec<f32>,
    resid_out_ref: Vec<f32>,
    core_ref: Vec<f32>,
    planes_ref: Planes,
    rows_after_gather: Vec<(Vec<f32>, Vec<f32>)>,
    mix_out: Vec<Vec<f32>>,
    resid_out: Vec<Vec<f32>>,
    core: Vec<Vec<f32>>,
    planes: Vec<Planes>,
    head_runs: Vec<std::ops::Range<usize>>,
    words: Vec<u32>,
}

type Planes = (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>);

fn run_rig(name: &'static str) -> Result<Evidence, EngError> {
    let d = step_dims();
    let inv = inventory();
    let options = options();
    let table_layers = LayerTable::new(&d, &inv)?;
    let mut layer = None;
    for l in 0..table_layers.len() {
        if table_layers.slot(l)?.kind == AttentionKind::Full {
            layer = Some(l);
            break;
        }
    }
    let layer = layer.ok_or_else(|| EngError::Checkpoint("no full-attention layer".into()))?;
    let ordinal = table_layers.slot(layer)?.ordinal;
    let resid_slot = if layer.is_multiple_of(2) {
        Slot::ResidA
    } else {
        Slot::ResidB
    };

    // The shared fixture: one Q4_K source per dense role of the FULL kind,
    // one deterministic raw row per raw spec, one residual row.
    let geometry = MixerGeometry::new(&d)?;
    let mut rng = Rng::new(0xC0FFEE);
    let dense_specs = geometry.dense_specs(WeightLayout::Native, AttentionKind::Full)?;
    let sources: Vec<(Role, Vec<u8>, usize, usize)> = dense_specs
        .iter()
        .map(|spec| {
            (
                spec.role,
                q4k_rows(spec.rows, spec.k, &mut rng),
                spec.rows,
                spec.k,
            )
        })
        .collect();
    let raw_specs = geometry.raw_specs(AttentionKind::Full)?;
    let raws: Vec<(Role, Vec<f32>)> = raw_specs
        .iter()
        .map(|spec| (spec.role, raw_values(&spec.role, spec.values)))
        .collect();
    let resid: Vec<f32> = (0..d.shape.hidden)
        .map(|i| ((i * 37) % 199) as f32 / 199.0 - 0.5)
        .collect();

    let admission = served_admission();
    let (stack, mut states) = CardStack::new(&d, &options, &[0, 0], &admission)?;
    let ctx0 = &stack.contexts()[0];
    let stream0 = &stack.streams()[0];

    // ---- The reference: the single-device fused block on card 0's context.
    let scratch = KernelScratch::for_dims(&d, 1, options.kv_mode)?;
    let plan_ref = StepPlan::new(&d, 1, &options, &scratch)?;
    let arena_ref = ctx0.arena(&plan_ref.arena_tensors())?;
    let mut state_ref = ResidentState::allocate(ctx0, stream0, &d, &options)?;
    state_ref.write_step_params(stream0, &StepParams::new(0, 0)?)?;
    let write_row =
        |stream: &crate::kernels::cuda::DeviceStream,
         view: &crate::kernels::cuda::DeviceRange<'_>,
         bytes: &[u8]|
         -> Result<(), EngError> { view.buffer().write_at(stream, view.offset(), bytes) };
    write_row(
        stream0,
        &arena_ref.tensor(
            plan_ref
                .index(resid_slot)
                .ok_or_else(|| EngError::Checkpoint("resid".into()))?,
        )?,
        &f32_bytes(&resid),
    )?;
    for slot in [Slot::Q8Mix, Slot::Q8DeltaOut, Slot::Q8AttnOut] {
        if let Some(at) = plan_ref.index(slot) {
            let view = arena_ref.tensor(at)?;
            write_row(stream0, &view, &vec![0u8; view.bytes()])?;
        }
    }
    let natives_ref: Vec<NativeWeight> = sources
        .iter()
        .map(|(_, source, rows, k)| upload_native(ctx0, source, *rows, *k))
        .collect::<Result<_, _>>()?;
    let dense_ref: Vec<(Role, DenseWeight<'_>)> = dense_specs
        .iter()
        .zip(&natives_ref)
        .map(|(spec, native)| (spec.role, DenseWeight::Native(native)))
        .collect();
    let raw_bufs_ref: Vec<(Role, DeviceBuffer)> = raws
        .iter()
        .map(|(role, values)| Ok((*role, ctx0.upload(&f32_bytes(values))?)))
        .collect::<Result<_, EngError>>()?;
    let raw_ref: Vec<(Role, crate::kernels::cuda::DeviceRange<'_>)> = raw_bufs_ref
        .iter()
        .map(|(role, buf)| Ok((*role, buf.range(0, buf.bytes())?)))
        .collect::<Result<_, EngError>>()?;
    let mut entries: Vec<Option<LayerWeights<'_>>> =
        (0..table_layers.len()).map(|_| None).collect();
    entries[layer] = Some(LayerWeights::for_tests(dense_ref, raw_ref));
    let parts = MixerParts {
        geometry,
        table: &table_layers,
        weights: MixerWeights::from_layers(entries),
        views: MixerViews::new(
            &plan_ref,
            &arena_ref,
            &geometry,
            WeightLayout::Native,
            options.kv_mode,
        )?,
        layout: WeightLayout::Native,
        kv_mode: options.kv_mode,
        paged: None,
        sparse: None,
    };
    let mixers_ref = Mixers::new(ctx0, parts)?;
    let resid_view_ref = arena_ref.tensor(
        plan_ref
            .index(resid_slot)
            .ok_or_else(|| EngError::Checkpoint("resid".into()))?,
    )?;
    mixers_ref.run_block(
        stream0,
        &state_ref,
        layer,
        BlockIo {
            resid: resid_view_ref,
            admit_pos: 0,
        },
    )?;
    stream0.synchronize()?;
    let read_slot = |slot: Slot| -> Result<Vec<f32>, EngError> {
        let view = arena_ref.tensor(
            plan_ref
                .index(slot)
                .ok_or_else(|| EngError::Checkpoint(format!("no {slot:?}")))?,
        )?;
        let mut bytes = vec![0u8; view.bytes()];
        view.buffer().read_at(stream0, view.offset(), &mut bytes)?;
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect())
    };
    let read_plane = |plane: Plane| -> Result<Vec<u8>, EngError> {
        let view = state_ref.kv_plane(ordinal, plane)?;
        let mut bytes = vec![0u8; view.bytes()];
        view.buffer().read_at(stream0, view.offset(), &mut bytes)?;
        Ok(bytes)
    };
    let mix_out_ref = read_slot(Slot::MixOut)?;
    let resid_out_ref = read_slot(resid_slot)?;
    let core_ref = read_slot(Slot::AttnCore)?;
    let planes_ref = (
        read_plane(Plane::K)?,
        read_plane(Plane::KScales)?,
        read_plane(Plane::V)?,
        read_plane(Plane::VScales)?,
    );

    // ---- The distributed block: per-card sliced weights, the card op
    // list, the `CardStack` execution.
    let table = ShardTable::new(&d.shape, 2, 32)?;
    for (card, state) in states.iter_mut().enumerate() {
        state.write_step_params(&stack.streams()[card], &StepParams::new(0, 0)?)?;
        let stream = &stack.streams()[card];
        let resid_at = stack.plans()[card]
            .index(resid_slot)
            .ok_or_else(|| EngError::Checkpoint("resid".into()))?;
        write_row(
            stream,
            &stack.shards()[card].tensor(resid_at)?,
            &f32_bytes(&resid),
        )?;
        for slot in [Slot::Q8Mix, Slot::Q8AttnOut] {
            if let Some(at) = stack.plans()[card].index(slot) {
                let view = stack.shards()[card].tensor(at)?;
                write_row(stream, &view, &vec![0u8; view.bytes()])?;
            }
        }
    }
    let kv_runs = [table.kv_rows(0)?, table.kv_rows(1)?];
    let q_rows = [table.attn_q_rows(0)?, table.attn_q_rows(1)?];
    let o_cols = [table.attn_o_cols(0)?, table.attn_o_cols(1)?];
    let head_runs = [table.query_heads(0)?, table.query_heads(1)?];
    let mut natives: Vec<Vec<NativeWeight>> = Vec::with_capacity(2);
    let mut raw_bufs: Vec<Vec<(Role, DeviceBuffer)>> = Vec::with_capacity(2);
    for card in 0..2 {
        let g = MixerGeometry::for_card(&d, &table, card)?;
        let specs = g.dense_specs(WeightLayout::Native, AttentionKind::Full)?;
        let mut card_natives = Vec::with_capacity(specs.len());
        for spec in &specs {
            let (source, rows, k) = sources
                .iter()
                .find(|(role, _, _, _)| *role == spec.role)
                .map(|(_, s, r, k)| (s, *r, *k))
                .ok_or_else(|| EngError::Checkpoint(format!("no source for {:?}", spec.role)))?;
            let sliced: Vec<u8> = match spec.role {
                // Row-dealt projections: contiguous row blocks.
                Role::AttnQ => row_slice(source, k, q_rows[card].clone()),
                Role::AttnK | Role::AttnV => row_slice(source, k, kv_runs[card].clone()),
                // The out projection deals COLUMNS: a block-aligned k slice.
                Role::AttnOutput => {
                    let slice = plan_slice(
                        "rig",
                        GgufType::Q4K,
                        &[k as u64, rows as u64],
                        o_cols[card].start as u64..o_cols[card].end as u64,
                        0..rows as u64,
                    )?;
                    assemble_slice(source, &slice)?
                }
                other => {
                    return Err(EngError::Checkpoint(format!(
                        "unexpected dense role {other:?}"
                    )));
                }
            };
            card_natives.push(upload_native(
                &stack.contexts()[card],
                &sliced,
                spec.rows,
                spec.k,
            )?);
        }
        natives.push(card_natives);
        let mut card_raws = Vec::with_capacity(raws.len());
        for (role, values) in &raws {
            card_raws.push((*role, stack.contexts()[card].upload(&f32_bytes(values))?));
        }
        raw_bufs.push(card_raws);
    }
    let mut mixers: Vec<Mixers<'_>> = Vec::with_capacity(2);
    let mut ops_attn: Option<Vec<Vec<CardOp>>> = None;
    for card in 0..2 {
        let g = MixerGeometry::for_card(&d, &table, card)?;
        let specs = g.dense_specs(WeightLayout::Native, AttentionKind::Full)?;
        let dense: Vec<(Role, DenseWeight<'_>)> = specs
            .iter()
            .zip(&natives[card])
            .map(|(spec, native)| (spec.role, DenseWeight::Native(native)))
            .collect();
        let raw_views: Vec<(Role, crate::kernels::cuda::DeviceRange<'_>)> = raw_bufs[card]
            .iter()
            .map(|(role, buf)| Ok((*role, buf.range(0, buf.bytes())?)))
            .collect::<Result<_, EngError>>()?;
        let mut entries: Vec<Option<LayerWeights<'_>>> =
            (0..table_layers.len()).map(|_| None).collect();
        entries[layer] = Some(LayerWeights::for_tests(dense, raw_views));
        let ops = CardKindOps::new(&table, card, &g, options.kv_mode)?;
        let views = MixerViews::for_card(&stack.plans()[card], stack.shards()[card].arena(), &ops)?;
        ops_attn.get_or_insert_with(Vec::new).push(ops.attn.clone());
        let parts = MixerParts {
            geometry: g,
            table: &table_layers,
            weights: MixerWeights::from_layers(entries),
            views,
            layout: WeightLayout::Native,
            kv_mode: options.kv_mode,
            paged: None,
            sparse: None,
        };
        mixers.push(Mixers::new(&stack.contexts()[card], parts)?);
    }
    // The residual row enters through `BlockIo`: the caller holds the
    // views of the card plans' own resid slots.
    let resid_views: Vec<crate::kernels::cuda::DeviceRange<'_>> = (0..2)
        .map(|card| {
            let at = stack.plans()[card]
                .index(resid_slot)
                .ok_or_else(|| EngError::Checkpoint("resid".into()))?;
            stack.shards()[card].tensor(at)
        })
        .collect::<Result<_, EngError>>()?;
    let ops = ops_attn
        .as_ref()
        .ok_or_else(|| EngError::Checkpoint("ops".into()))?;
    let split = ops[0]
        .iter()
        .position(|op| matches!(op, super::mixer_ops::CardOp::GatherV))
        .ok_or_else(|| EngError::Checkpoint("no gather".into()))?
        + 1;
    let ops_gather: Vec<&[CardOp]> = ops.iter().map(|l| &l[..split]).collect();
    run_card_block(
        &stack,
        &states,
        &mixers,
        layer,
        ordinal,
        &ops_gather,
        &resid_views,
        0,
        &kv_runs,
    )?;
    let rows_after_gather = (0..2)
        .map(|card| {
            stack.streams()[card].synchronize()?;
            let read = |slot: Slot| -> Result<Vec<f32>, EngError> {
                let view = mixers[card].slot_view(slot)?;
                let mut bytes = vec![0u8; view.bytes()];
                view.buffer()
                    .read_at(&stack.streams()[card], view.offset(), &mut bytes)?;
                Ok(bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes(*c))
                    .collect())
            };
            Ok((read(Slot::KRows)?, read(Slot::VRows)?))
        })
        .collect::<Result<Vec<_>, EngError>>()?;
    // The second half runs the suffix after the gather boundary; the
    // rows it consumes were produced by the first half's gathers.
    let ops_suffix: Vec<&[CardOp]> = ops.iter().map(|l| &l[split..]).collect();
    run_card_block(
        &stack,
        &states,
        &mixers,
        layer,
        ordinal,
        &ops_suffix,
        &resid_views,
        0,
        &kv_runs,
    )?;
    for card in 0..2 {
        stack.streams()[card].synchronize()?;
    }

    // ---- Read back the card side.
    let read_card_slot = |card: usize, slot: Slot| -> Result<Vec<f32>, EngError> {
        let view = mixers[card].slot_view(slot)?;
        let mut bytes = vec![0u8; view.bytes()];
        view.buffer()
            .read_at(&stack.streams()[card], view.offset(), &mut bytes)?;
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect())
    };
    let read_card_plane = |card: usize, plane: Plane| -> Result<Vec<u8>, EngError> {
        let view = states[card].kv_plane(ordinal, plane)?;
        let mut bytes = vec![0u8; view.bytes()];
        view.buffer()
            .read_at(&stack.streams()[card], view.offset(), &mut bytes)?;
        Ok(bytes)
    };
    let mut words = Vec::with_capacity(2);
    for card in 0..2 {
        let mut word = [0u8; 4];
        stack.shards()[card]
            .ring_word()
            .read_at(&stack.streams()[card], 0, &mut word)?;
        words.push(u32::from_le_bytes(word));
    }
    let mix_out = (0..2)
        .map(|c| read_card_slot(c, Slot::MixOut))
        .collect::<Result<_, _>>()?;
    let resid_out = (0..2)
        .map(|c| {
            let at = stack.plans()[c]
                .index(resid_slot)
                .ok_or_else(|| EngError::Checkpoint("resid".into()))?;
            let view = stack.shards()[c].tensor(at)?;
            let mut bytes = vec![0u8; view.bytes()];
            view.buffer()
                .read_at(&stack.streams()[c], view.offset(), &mut bytes)?;
            Ok(bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect())
        })
        .collect::<Result<Vec<_>, EngError>>()?;
    let core = (0..2)
        .map(|c| read_card_slot(c, Slot::AttnCore))
        .collect::<Result<_, _>>()?;
    let mut planes = Vec::with_capacity(2);
    for card in 0..2 {
        planes.push((
            read_card_plane(card, Plane::K)?,
            read_card_plane(card, Plane::KScales)?,
            read_card_plane(card, Plane::V)?,
            read_card_plane(card, Plane::VScales)?,
        ));
    }
    Ok(Evidence {
        name,
        mix_out_ref,
        resid_out_ref,
        core_ref,
        planes_ref,
        rows_after_gather,
        mix_out,
        resid_out,
        core,
        planes,
        head_runs: head_runs.to_vec(),
        words,
    })
}

#[test]
fn a_full_attention_block_runs_distributed_and_matches_the_reference() {
    const NAME: &str = "a_full_attention_block_runs_distributed_and_matches_the_reference";
    if !device_present(NAME) {
        return;
    }
    let ev = run_rig(NAME).expect("the rig");
    // Exact: the KV cache planes (codes and scales) of the layer. The card's
    // planes hold only its own K/V heads (sliced at allocation), so the
    // reference's head-run slice is the expectation; the tail past the card's
    // heads is plane alignment padding, never compared.
    let full_heads = ev.head_runs[0].len() + ev.head_runs[1].len();
    for card in 0..2 {
        let got = &ev.planes[card];
        let want = &ev.planes_ref;
        for (plane, got_bytes, want_bytes) in [
            ("K codes", &got.0, &want.0),
            ("K scales", &got.1, &want.1),
            ("V codes", &got.2, &want.2),
            ("V scales", &got.3, &want.3),
        ] {
            let per_head = want_bytes.len() / full_heads;
            let start = ev.head_runs[card].start * per_head;
            let end = start + got_bytes.len().min(per_head * ev.head_runs[card].len());
            let want_slice = &want_bytes[start..end];
            let diff_count = got_bytes[..want_slice.len()]
                .iter()
                .zip(want_slice)
                .filter(|(g, w)| g != w)
                .count();
            println!(
                "{NAME}: card {card} {plane}: {diff_count} differing bytes over the card's {}                  head slice (plane holds {} B, per head {per_head} B)",
                ev.head_runs[card].len(),
                got_bytes.len(),
            );
            assert!(
                diff_count == 0,
                "card {card} {plane}: the appended cache must equal the reference's head slice"
            );
        }
    }
    // Exact: the gathered rows agree across cards, read back BEFORE the
    // append may touch the caches (later phases reuse the row arena, so a
    // post-block readback would compare clobbered scratch, not the rows).
    assert_eq!(
        ev.rows_after_gather[0].0, ev.rows_after_gather[1].0,
        "the gathered K row is bit-identical"
    );
    assert_eq!(
        ev.rows_after_gather[0].1, ev.rows_after_gather[1].1,
        "the gathered V row is bit-identical"
    );
    // Exact: the per-head decode against the reference's head slice.
    let head_stride = ev.core_ref.len() / (ev.head_runs[0].len() + ev.head_runs[1].len());
    for card in 0..2 {
        let core = &ev.core[card];
        let start = ev.head_runs[card].start * head_stride;
        let end = start + core.len();
        let slice = &ev.core_ref[start..end];
        let diff = max_abs_diff(core, slice);
        println!(
            "{NAME}: card {card} AttnCore max |diff| = {diff:.3e} over {} values",
            core.len()
        );
        assert_eq!(
            diff, 0.0,
            "card {card}: the per-head decode is expected exact"
        );
    }
    // Numerical: the summed mixer output and the residual epilogue.
    for card in 0..2 {
        let diff = max_abs_diff(&ev.mix_out[card], &ev.mix_out_ref);
        let bound = SPLIT_SUM_RELATIVE * max_abs(&ev.mix_out_ref).max(1.0);
        println!(
            "{NAME}: card {card} MixOut max |diff| = {diff:.3e} (bound {bound:.3e}, reference peak {})",
            max_abs(&ev.mix_out_ref)
        );
        assert!(
            diff <= bound,
            "card {card}: MixOut differs by {diff} > {bound}"
        );
        let diff = max_abs_diff(&ev.resid_out[card], &ev.resid_out_ref);
        let bound = SPLIT_SUM_RELATIVE * max_abs(&ev.resid_out_ref).max(1.0);
        println!("{NAME}: card {card} resid max |diff| = {diff:.3e} (bound {bound:.3e})");
        assert!(
            diff <= bound,
            "card {card}: the epilogue differs by {diff} > {bound}"
        );
    }
    for (card, word) in ev.words.iter().enumerate() {
        assert_eq!(*word, 0, "card {card}: ring clean");
    }
}

/// The DeltaNet block comparison, `?` inside; asserts live in the caller.
///
/// The card sharding of DeltaNet is BY VALUE HEAD: every card owns its
/// heads' rows of the four in-projections, its value-head rows of the gate
/// and beta/alpha scalars, its channel segments of the conv state, and its
/// column block of the out projection. The recurrent state is LOCAL to a
/// card by construction - no gathers - so the card list is the single-
/// device list over the card geometry plus the one `MixerSum` marker.
struct DeltaEvidence {
    mix_out_ref: Vec<Vec<f32>>,
    resid_out_ref: Vec<Vec<f32>>,
    mix_out: Vec<Vec<Vec<f32>>>,
    resid_out: Vec<Vec<Vec<f32>>>,
    /// Last token's DeltaCore per card and the reference row.
    core: Vec<Vec<f32>>,
    core_ref: Vec<f32>,
    /// Recurrent + convolution state after token 0 (bit-exact expected)
    /// and after the last token (numerical: the residual the tokens 1..
    /// consume already carries the split-sum drift).
    rec0_ref: Vec<f32>,
    rec_ref: Vec<f32>,
    conv0_ref: Vec<f32>,
    conv_ref: Vec<f32>,
    rec0: Vec<Vec<f32>>,
    rec: Vec<Vec<f32>>,
    conv0: Vec<Vec<f32>>,
    conv: Vec<Vec<f32>>,
    /// The card's value-head run and conv channel segments into the
    /// reference rows.
    head_runs: Vec<std::ops::Range<usize>>,
    conv_segments: Vec<Vec<std::ops::Range<usize>>>,
    /// Per-head strides: the delta core is [heads][value_dim], the
    /// recurrent state [heads][key_dim x value_dim], the conv state
    /// [channel][conv_kernel-1].
    core_stride: usize,
    rec_stride: usize,
    conv_taps: usize,
    words: Vec<u32>,
}

fn run_delta_rig(name: &'static str) -> Result<DeltaEvidence, EngError> {
    const TOKENS: usize = 3;
    let d = step_dims();
    let inv = inventory();
    let options = options();
    let table_layers = LayerTable::new(&d, &inv)?;
    let mut layer = None;
    for l in 0..table_layers.len() {
        if table_layers.slot(l)?.kind == AttentionKind::DeltaNet {
            layer = Some(l);
            break;
        }
    }
    let layer = layer.ok_or_else(|| EngError::Checkpoint("no DeltaNet layer".into()))?;
    let ordinal = table_layers.slot(layer)?.ordinal;
    let resid_slot = if layer.is_multiple_of(2) {
        Slot::ResidA
    } else {
        Slot::ResidB
    };

    // The shared fixture over the FULL geometry: one Q4_K source per dense
    // role of the DELTA kind, one deterministic raw row per raw spec, one
    // residual row.
    let geometry = MixerGeometry::new(&d)?;
    // The state layout strides come from the HOST shape (the geometry's
    // delta view carries row offsets, not the head dims).
    let shape_delta = &d.shape.delta;
    let head_block = shape_delta.key_dim * shape_delta.value_dim;
    let conv_taps = shape_delta.conv_kernel - 1;
    let mut rng = Rng::new(0xDE17A5);
    let dense_specs = geometry.dense_specs(WeightLayout::Native, AttentionKind::DeltaNet)?;
    let sources: Vec<(Role, Vec<u8>, usize, usize)> = dense_specs
        .iter()
        .map(|spec| {
            (
                spec.role,
                q4k_rows(spec.rows, spec.k, &mut rng),
                spec.rows,
                spec.k,
            )
        })
        .collect();
    let raw_specs = geometry.raw_specs(AttentionKind::DeltaNet)?;
    let raws: Vec<(Role, Vec<f32>)> = raw_specs
        .iter()
        .map(|spec| (spec.role, raw_values(&spec.role, spec.values)))
        .collect();
    let resid: Vec<f32> = (0..d.shape.hidden)
        .map(|i| ((i * 41) % 211) as f32 / 211.0 - 0.5)
        .collect();

    let admission = served_admission();
    let (stack, mut states) = CardStack::new(&d, &options, &[0, 0], &admission)?;
    let ctx0 = &stack.contexts()[0];
    let stream0 = &stack.streams()[0];

    // ---- The reference: the single-device DeltaNet block, TOKENS tokens,
    // each reading the residual the previous token's epilogue wrote.
    let scratch = KernelScratch::for_dims(&d, 1, options.kv_mode)?;
    let plan_ref = StepPlan::new(&d, 1, &options, &scratch)?;
    let arena_ref = ctx0.arena(&plan_ref.arena_tensors())?;
    let mut state_ref = ResidentState::allocate(ctx0, stream0, &d, &options)?;
    let write_row =
        |stream: &crate::kernels::cuda::DeviceStream,
         view: &crate::kernels::cuda::DeviceRange<'_>,
         bytes: &[u8]|
         -> Result<(), EngError> { view.buffer().write_at(stream, view.offset(), bytes) };
    write_row(
        stream0,
        &arena_ref.tensor(
            plan_ref
                .index(resid_slot)
                .ok_or_else(|| EngError::Checkpoint("resid".into()))?,
        )?,
        &f32_bytes(&resid),
    )?;
    for slot in [Slot::Q8Mix, Slot::Q8DeltaOut, Slot::Q8AttnOut] {
        if let Some(at) = plan_ref.index(slot) {
            let view = arena_ref.tensor(at)?;
            write_row(stream0, &view, &vec![0u8; view.bytes()])?;
        }
    }
    let natives_ref: Vec<NativeWeight> = sources
        .iter()
        .map(|(_, source, rows, k)| upload_native(ctx0, source, *rows, *k))
        .collect::<Result<_, _>>()?;
    let dense_ref: Vec<(Role, DenseWeight<'_>)> = dense_specs
        .iter()
        .zip(&natives_ref)
        .map(|(spec, native)| (spec.role, DenseWeight::Native(native)))
        .collect();
    let raw_bufs_ref: Vec<(Role, DeviceBuffer)> = raws
        .iter()
        .map(|(role, values)| Ok((*role, ctx0.upload(&f32_bytes(values))?)))
        .collect::<Result<_, EngError>>()?;
    let raw_ref: Vec<(Role, crate::kernels::cuda::DeviceRange<'_>)> = raw_bufs_ref
        .iter()
        .map(|(role, buf)| Ok((*role, buf.range(0, buf.bytes())?)))
        .collect::<Result<_, EngError>>()?;
    let mut entries: Vec<Option<LayerWeights<'_>>> =
        (0..table_layers.len()).map(|_| None).collect();
    entries[layer] = Some(LayerWeights::for_tests(dense_ref, raw_ref));
    let parts = MixerParts {
        geometry,
        table: &table_layers,
        weights: MixerWeights::from_layers(entries),
        views: MixerViews::new(
            &plan_ref,
            &arena_ref,
            &geometry,
            WeightLayout::Native,
            options.kv_mode,
        )?,
        layout: WeightLayout::Native,
        kv_mode: options.kv_mode,
        paged: None,
        sparse: None,
    };
    let mixers_ref = Mixers::new(ctx0, parts)?;
    let read_ref_slot = |slot: Slot| -> Result<Vec<f32>, EngError> {
        let view = arena_ref.tensor(
            plan_ref
                .index(slot)
                .ok_or_else(|| EngError::Checkpoint(format!("no {slot:?}")))?,
        )?;
        let mut bytes = vec![0u8; view.bytes()];
        view.buffer().read_at(stream0, view.offset(), &mut bytes)?;
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect())
    };
    let read_ref_state = |state: &ResidentState| -> Result<(Vec<f32>, Vec<f32>), EngError> {
        let rec = state.delta_rec(0, ordinal)?;
        let conv = state.delta_conv(0, ordinal)?;
        let read = |view: crate::kernels::cuda::DeviceRange<'_>| -> Result<Vec<f32>, EngError> {
            let mut bytes = vec![0u8; view.bytes()];
            view.buffer().read_at(stream0, view.offset(), &mut bytes)?;
            Ok(bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect())
        };
        Ok((read(rec)?, read(conv)?))
    };
    let mut mix_out_ref = Vec::with_capacity(TOKENS);
    let mut resid_out_ref = Vec::with_capacity(TOKENS);
    let mut rec0_ref = None;
    let resid_at_ref = plan_ref
        .index(resid_slot)
        .ok_or_else(|| EngError::Checkpoint("resid".into()))?;
    for token in 0..TOKENS {
        // The book pins the position to the committed rows; a DeltaNet block
        // commits none, so the position stays 0 (the delta ops read no
        // position - only the attention prep does).
        state_ref.write_step_params(stream0, &StepParams::new(token as i32 + 1, 0)?)?;
        mixers_ref.run_block(
            stream0,
            &state_ref,
            layer,
            BlockIo {
                resid: arena_ref.tensor(resid_at_ref)?,
                admit_pos: token,
            },
        )?;
        stream0.synchronize()?;
        mix_out_ref.push(read_ref_slot(Slot::MixOut)?);
        resid_out_ref.push(read_ref_slot(resid_slot)?);
        if token == 0 {
            rec0_ref = Some(read_ref_state(&state_ref)?);
        }
    }
    let (rec0_ref, conv0_ref) = rec0_ref.ok_or_else(|| EngError::Checkpoint("no state".into()))?;
    let (rec_ref, conv_ref) = read_ref_state(&state_ref)?;
    let core_ref = read_ref_slot(Slot::DeltaCore)?;

    // ---- The distributed block: per-card sliced weights (the shard
    // table's own dealing), the card op list, the `CardStack` execution.
    let table = ShardTable::new(&d.shape, 2, 32)?;
    for (card, state) in states.iter_mut().enumerate() {
        state.write_step_params(&stack.streams()[card], &StepParams::new(0, 0)?)?;
        let stream = &stack.streams()[card];
        let resid_at = stack.plans()[card]
            .index(resid_slot)
            .ok_or_else(|| EngError::Checkpoint("resid".into()))?;
        write_row(
            stream,
            &stack.shards()[card].tensor(resid_at)?,
            &f32_bytes(&resid),
        )?;
        for slot in [Slot::Q8Mix, Slot::Q8DeltaOut, Slot::Q8AttnOut] {
            if let Some(at) = stack.plans()[card].index(slot) {
                let view = stack.shards()[card].tensor(at)?;
                write_row(stream, &view, &vec![0u8; view.bytes()])?;
            }
        }
    }
    let head_runs = [table.value_heads(0)?, table.value_heads(1)?];
    let conv_segments: Vec<Vec<std::ops::Range<usize>>> = (0..2)
        .map(|card| Ok(table.delta_qkv_segments(card)?.to_vec()))
        .collect::<Result<_, EngError>>()?;
    let mut natives: Vec<Vec<NativeWeight>> = Vec::with_capacity(2);
    let mut raw_bufs: Vec<Vec<(Role, DeviceBuffer)>> = Vec::with_capacity(2);
    for card in 0..2 {
        let g = MixerGeometry::for_card(&d, &table, card)?;
        let specs = g.dense_specs(WeightLayout::Native, AttentionKind::DeltaNet)?;
        let mut card_natives = Vec::with_capacity(specs.len());
        for spec in &specs {
            let (source, rows, k) = sources
                .iter()
                .find(|(role, _, _, _)| *role == spec.role)
                .map(|(_, s, r, k)| (s, *r, *k))
                .ok_or_else(|| EngError::Checkpoint(format!("no source for {:?}", spec.role)))?;
            let sliced: Vec<u8> = match spec.role {
                // The qkv projection and the conv weight deal CHANNEL rows
                // in three per-head segments (q, k, v), not one run.
                Role::AttnQkv => conv_segments[card]
                    .iter()
                    .flat_map(|seg| row_slice(source, k, seg.clone()))
                    .collect(),
                Role::AttnGate => row_slice(source, k, table.delta_z_rows(card)?),
                Role::SsmBeta | Role::SsmAlpha => row_slice(source, k, table.value_heads(card)?),
                // The out projection deals COLUMNS: a block-aligned k slice.
                Role::SsmOut => {
                    let slice = plan_slice(
                        "rig",
                        GgufType::Q4K,
                        &[k as u64, rows as u64],
                        table.delta_out_cols(card)?.start as u64
                            ..table.delta_out_cols(card)?.end as u64,
                        0..rows as u64,
                    )?;
                    assemble_slice(source, &slice)?
                }
                other => {
                    return Err(EngError::Checkpoint(format!(
                        "unexpected dense role {other:?}"
                    )));
                }
            };
            card_natives.push(upload_native(
                &stack.contexts()[card],
                &sliced,
                spec.rows,
                spec.k,
            )?);
        }
        natives.push(card_natives);
        let mut card_raws = Vec::with_capacity(raws.len());
        for (role, values) in &raws {
            let sliced: Vec<f32> = match role {
                // Per value head: one scalar each.
                Role::SsmA | Role::SsmDtBias => values[table.value_heads(card)?].to_vec(),
                // Per conv channel: one conv_kernel-wide row each, dealt in
                // the same three channel segments as the qkv rows.
                Role::SsmConv1d => conv_segments[card]
                    .iter()
                    .flat_map(|seg| {
                        values
                            [seg.start * shape_delta.conv_kernel..seg.end * shape_delta.conv_kernel]
                            .to_vec()
                    })
                    .collect(),
                // Whole: the norms and the mixer norms.
                _ => values.clone(),
            };
            card_raws.push((*role, stack.contexts()[card].upload(&f32_bytes(&sliced))?));
        }
        raw_bufs.push(card_raws);
    }
    let mut mixers: Vec<Mixers<'_>> = Vec::with_capacity(2);
    let mut ops_delta: Option<Vec<Vec<CardOp>>> = None;
    for card in 0..2 {
        let g = MixerGeometry::for_card(&d, &table, card)?;
        let specs = g.dense_specs(WeightLayout::Native, AttentionKind::DeltaNet)?;
        let dense: Vec<(Role, DenseWeight<'_>)> = specs
            .iter()
            .zip(&natives[card])
            .map(|(spec, native)| (spec.role, DenseWeight::Native(native)))
            .collect();
        let raw_views: Vec<(Role, crate::kernels::cuda::DeviceRange<'_>)> = raw_bufs[card]
            .iter()
            .map(|(role, buf)| Ok((*role, buf.range(0, buf.bytes())?)))
            .collect::<Result<_, EngError>>()?;
        let mut entries: Vec<Option<LayerWeights<'_>>> =
            (0..table_layers.len()).map(|_| None).collect();
        entries[layer] = Some(LayerWeights::for_tests(dense, raw_views));
        let ops = CardKindOps::new(&table, card, &g, options.kv_mode)?;
        let views = MixerViews::for_card(&stack.plans()[card], stack.shards()[card].arena(), &ops)?;
        ops_delta
            .get_or_insert_with(Vec::new)
            .push(ops.delta.clone());
        let parts = MixerParts {
            geometry: g,
            table: &table_layers,
            weights: MixerWeights::from_layers(entries),
            views,
            layout: WeightLayout::Native,
            kv_mode: options.kv_mode,
            paged: None,
            sparse: None,
        };
        mixers.push(Mixers::new(&stack.contexts()[card], parts)?);
    }
    let resid_views: Vec<crate::kernels::cuda::DeviceRange<'_>> = (0..2)
        .map(|card| {
            let at = stack.plans()[card]
                .index(resid_slot)
                .ok_or_else(|| EngError::Checkpoint("resid".into()))?;
            stack.shards()[card].tensor(at)
        })
        .collect::<Result<_, EngError>>()?;
    let ops = ops_delta
        .as_ref()
        .ok_or_else(|| EngError::Checkpoint("ops".into()))?;
    let lists: Vec<&[CardOp]> = ops.iter().map(|l| l.as_slice()).collect();
    let kv_runs = [table.kv_rows(0)?, table.kv_rows(1)?];
    let read_card_slot = |card: usize, slot: Slot| -> Result<Vec<f32>, EngError> {
        let view = mixers[card].slot_view(slot)?;
        let mut bytes = vec![0u8; view.bytes()];
        view.buffer()
            .read_at(&stack.streams()[card], view.offset(), &mut bytes)?;
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect())
    };
    let read_card_state = |card: usize| -> Result<(Vec<f32>, Vec<f32>), EngError> {
        let stream = &stack.streams()[card];
        let state = &states[card];
        let rec = state.delta_rec(0, ordinal)?;
        let conv = state.delta_conv(0, ordinal)?;
        let read = |view: crate::kernels::cuda::DeviceRange<'_>| -> Result<Vec<f32>, EngError> {
            let mut bytes = vec![0u8; view.bytes()];
            view.buffer().read_at(stream, view.offset(), &mut bytes)?;
            Ok(bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect())
        };
        Ok((read(rec)?, read(conv)?))
    };
    let mut mix_out = Vec::with_capacity(TOKENS);
    let mut resid_out = Vec::with_capacity(TOKENS);
    let mut rec0 = None;
    for token in 0..TOKENS {
        run_card_block(
            &stack,
            &states,
            &mixers,
            layer,
            ordinal,
            &lists,
            &resid_views,
            token,
            &kv_runs,
        )?;
        for card in 0..2 {
            stack.streams()[card].synchronize()?;
        }
        let per_card_mix = (0..2)
            .map(|card| read_card_slot(card, Slot::MixOut))
            .collect::<Result<_, EngError>>()?;
        let per_card_resid = (0..2)
            .map(|card| {
                let at = stack.plans()[card]
                    .index(resid_slot)
                    .ok_or_else(|| EngError::Checkpoint("resid".into()))?;
                let view = stack.shards()[card].tensor(at)?;
                let mut bytes = vec![0u8; view.bytes()];
                view.buffer()
                    .read_at(&stack.streams()[card], view.offset(), &mut bytes)?;
                Ok(bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes(*c))
                    .collect())
            })
            .collect::<Result<_, EngError>>()?;
        mix_out.push(per_card_mix);
        resid_out.push(per_card_resid);
        if token == 0 {
            rec0 = Some(
                (0..2)
                    .map(read_card_state)
                    .collect::<Result<Vec<_>, EngError>>()?,
            );
        }
    }
    let rec0 = rec0.ok_or_else(|| EngError::Checkpoint("no state".into()))?;
    let (rec, conv): (Vec<_>, Vec<_>) = (0..2)
        .map(read_card_state)
        .collect::<Result<Vec<_>, EngError>>()?
        .into_iter()
        .unzip();
    let core = (0..2)
        .map(|card| read_card_slot(card, Slot::DeltaCore))
        .collect::<Result<_, EngError>>()?;
    let mut words = Vec::with_capacity(2);
    for card in 0..2 {
        let mut word = [0u8; 4];
        stack.shards()[card]
            .ring_word()
            .read_at(&stack.streams()[card], 0, &mut word)?;
        words.push(u32::from_le_bytes(word));
    }
    println!(
        "{name}: {TOKENS} tokens, heads {}+{}, conv channels {}",
        head_runs[0].len(),
        head_runs[1].len(),
        conv_segments[0].iter().map(|s| s.len()).sum::<usize>(),
    );
    Ok(DeltaEvidence {
        mix_out_ref,
        resid_out_ref,
        mix_out,
        resid_out,
        core,
        core_ref,
        rec0_ref,
        rec_ref,
        conv0_ref,
        conv_ref,
        rec0: rec0.iter().map(|(r, _)| r.clone()).collect(),
        conv0: rec0.iter().map(|(_, c)| c.clone()).collect(),
        rec,
        conv,
        head_runs: head_runs.to_vec(),
        conv_segments,
        core_stride: shape_delta.value_dim,
        rec_stride: head_block,
        conv_taps,
        words,
    })
}

#[test]
fn a_deltanet_block_runs_distributed_and_matches_the_reference() {
    const NAME: &str = "a_deltanet_block_runs_distributed_and_matches_the_reference";
    if !device_present(NAME) {
        return;
    }
    let ev = run_delta_rig(NAME).expect("the rig");
    let (core_stride, rec_stride, conv_taps) = (ev.core_stride, ev.rec_stride, ev.conv_taps);
    // Exact after token 0: the whole per-card pipeline is local kernels
    // over exactly-sliced weights and the same uploaded residual, so the
    // card's DeltaNet state must equal the reference's head blocks bit for
    // bit before any split-sum drift feeds back through the residual.
    for card in 0..2 {
        let run = &ev.head_runs[card];
        let want = &ev.rec0_ref[run.start * rec_stride..run.end * rec_stride];
        let got = &ev.rec0[card][..want.len()];
        if got != want {
            let first = got
                .iter()
                .zip(want)
                .position(|(a, b)| a.to_bits() != b.to_bits())
                .unwrap_or(want.len());
            let per_head: Vec<f32> = got
                .chunks(rec_stride)
                .zip(want.chunks(rec_stride))
                .map(|(a, b)| max_abs_diff(a, b))
                .collect();
            println!(
                "{NAME}: card {card} rec0 first diff at {first} (head block {}), per-head max diffs {:?}",
                first / rec_stride,
                &per_head[..per_head.len().min(16)]
            );
        }
        assert_eq!(
            got, want,
            "card {card}: the recurrent state after token 0 is bit-exact"
        );
        // The conv state is TAP-major ([taps][channels]); the card's local
        // channels are its segments in stored order.
        let full_channels = ev.conv0_ref.len() / conv_taps;
        let mut conv_want = Vec::new();
        for tap in 0..conv_taps {
            for seg in &ev.conv_segments[card] {
                conv_want.extend_from_slice(
                    &ev.conv0_ref[tap * full_channels + seg.start..tap * full_channels + seg.end],
                );
            }
        }
        assert_eq!(
            &ev.conv0[card][..conv_want.len()],
            conv_want,
            "card {card}: the conv state after token 0 is bit-exact"
        );
    }
    // Exact on the last token: the per-head decode-equivalent (the delta
    // core over the card's own heads) stays exact - it never reads the
    // summed row - but its INPUTS carry the residual drift from tokens
    // 1.., so this is a tight numerical comparison instead.
    for card in 0..2 {
        let run = &ev.head_runs[card];
        let start = run.start * core_stride;
        let slice = &ev.core_ref[start..start + ev.core[card].len()];
        let diff = max_abs_diff(&ev.core[card], slice);
        let bound = SPLIT_SUM_RELATIVE * max_abs(slice).max(1.0);
        println!("{NAME}: card {card} DeltaCore max |diff| = {diff:.3e} (bound {bound:.3e})");
        assert!(
            diff <= bound,
            "card {card}: DeltaCore differs by {diff} > {bound}"
        );
        let full_channels = ev.conv_ref.len() / conv_taps;
        let mut conv_want = Vec::new();
        for tap in 0..conv_taps {
            for seg in &ev.conv_segments[card] {
                conv_want.extend_from_slice(
                    &ev.conv_ref[tap * full_channels + seg.start..tap * full_channels + seg.end],
                );
            }
        }
        let diff = max_abs_diff(&ev.conv[card][..conv_want.len()], &conv_want);
        let bound = SPLIT_SUM_RELATIVE * max_abs(&conv_want).max(1.0);
        println!("{NAME}: card {card} conv state max |diff| = {diff:.3e} (bound {bound:.3e})");
        assert!(
            diff <= bound,
            "card {card}: the conv state differs by {diff} > {bound}"
        );
        let want = &ev.rec_ref[run.start * rec_stride..run.end * rec_stride];
        let diff = max_abs_diff(&ev.rec[card][..want.len()], want);
        let bound = SPLIT_SUM_RELATIVE * max_abs(want).max(1.0);
        println!("{NAME}: card {card} recurrent state max |diff| = {diff:.3e} (bound {bound:.3e})");
        assert!(
            diff <= bound,
            "card {card}: the recurrent state differs by {diff} > {bound}"
        );
    }
    // Numerical on every token: the summed mixer output (MixerSum of the
    // column-dealt out projection) and the residual epilogue.
    for token in 0..ev.mix_out.len() {
        for card in 0..2 {
            let diff = max_abs_diff(&ev.mix_out[token][card], &ev.mix_out_ref[token]);
            let bound = SPLIT_SUM_RELATIVE * max_abs(&ev.mix_out_ref[token]).max(1.0);
            println!(
                "{NAME}: token {token} card {card} MixOut max |diff| = {diff:.3e} (bound {bound:.3e})"
            );
            assert!(
                diff <= bound,
                "token {token} card {card}: MixOut differs by {diff} > {bound}"
            );
            let diff = max_abs_diff(&ev.resid_out[token][card], &ev.resid_out_ref[token]);
            let bound = SPLIT_SUM_RELATIVE * max_abs(&ev.resid_out_ref[token]).max(1.0);
            assert!(
                diff <= bound,
                "token {token} card {card}: the epilogue differs by {diff} > {bound}"
            );
        }
    }
    for (card, word) in ev.words.iter().enumerate() {
        assert_eq!(*word, 0, "card {card}: ring clean");
    }
}

/// One tensor of a test GGUF file. A second minimal copy of the writer the
/// projector tests carry (`family/project/cuda/tests.rs`); extract both on
/// the third use.
#[derive(Clone)]
pub(super) struct FileEntry {
    pub(super) name: String,
    pub(super) ty: GgufType,
    pub(super) dims: Vec<u64>,
    pub(super) payload: Vec<u8>,
}

/// A GGUF v3 file with no metadata; payloads 32-byte aligned like the
/// projector fixture writer.
fn gguf_file_bytes(entries: &[FileEntry]) -> Vec<u8> {
    const ALIGN: usize = 32;
    let mut header = b"GGUF".to_vec();
    header.extend_from_slice(&3u32.to_le_bytes());
    header.extend_from_slice(&(entries.len() as u64).to_le_bytes());
    header.extend_from_slice(&0u64.to_le_bytes());
    let mut data: Vec<u8> = Vec::new();
    for entry in entries {
        header.extend_from_slice(&(entry.name.len() as u64).to_le_bytes());
        header.extend_from_slice(entry.name.as_bytes());
        header.extend_from_slice(&(entry.dims.len() as u32).to_le_bytes());
        for &dim in &entry.dims {
            header.extend_from_slice(&dim.to_le_bytes());
        }
        header.extend_from_slice(&entry.ty.id().to_le_bytes());
        data.resize(data.len().next_multiple_of(ALIGN), 0);
        header.extend_from_slice(&(data.len() as u64).to_le_bytes());
        data.extend_from_slice(&entry.payload);
    }
    header.resize(header.len().next_multiple_of(ALIGN), 0);
    header.extend_from_slice(&data);
    header
}

pub(super) fn write_gguf(
    dir: &std::path::Path,
    file: &str,
    entries: &[FileEntry],
) -> std::path::PathBuf {
    let path = dir.join(file);
    std::fs::write(&path, gguf_file_bytes(entries)).expect("write the fixture file");
    path
}

/// The real-file variant of [`run_rig`]: the SAME fixture bytes, but
/// written into a GGUF-v3 file, opened through the production reader,
/// sliced by the planner-selected manifests, uploaded by
/// `load_card_weights`, and resolved per card by the production
/// `MixerWeights::new` (the name+shape check runs against the CARD's
/// geometry, so a wrongly sliced tensor refuses by name) - then run
/// through the same `CardStack` launchers. The projector's context is
/// dropped before the stack takes its two leases; the loaded `CudaWeights`
/// hold their own device leases and survive it.
fn run_gguf_rig(name: &'static str) -> Result<Evidence, EngError> {
    use crate::family::project::cuda::{BorrowedLoader, LoadOptions};
    use crate::family::qwen36::cuda::card_load::{CardLoad, card_loads};
    use crate::family::qwen36::cuda::card_weights::load_card_weights;
    use crate::weights::{GgufLimits, GgufWeights};

    let d = step_dims();
    let inv = inventory();
    let options = options();
    let table_layers = LayerTable::new(&d, &inv)?;
    let mut layer = None;
    for l in 0..table_layers.len() {
        if table_layers.slot(l)?.kind == AttentionKind::Full {
            layer = Some(l);
            break;
        }
    }
    let layer = layer.ok_or_else(|| EngError::Checkpoint("no full-attention layer".into()))?;
    let ordinal = table_layers.slot(layer)?.ordinal;
    let resid_slot = if layer.is_multiple_of(2) {
        Slot::ResidA
    } else {
        Slot::ResidB
    };

    // The same deterministic fixture as run_rig, kept as FILE entries.
    let geometry = MixerGeometry::new(&d)?;
    let mut rng = Rng::new(0xC0FFEE);
    let dense_specs = geometry.dense_specs(WeightLayout::Native, AttentionKind::Full)?;
    let raw_specs = geometry.raw_specs(AttentionKind::Full)?;
    let mut entries: Vec<FileEntry> = Vec::new();
    for spec in &dense_specs {
        entries.push(FileEntry {
            name: table_layers.name(layer, spec.role)?.to_owned(),
            ty: GgufType::Q4K,
            dims: vec![spec.k as u64, spec.rows as u64],
            payload: q4k_rows(spec.rows, spec.k, &mut rng),
        });
    }
    for spec in &raw_specs {
        entries.push(FileEntry {
            name: table_layers.name(layer, spec.role)?.to_owned(),
            ty: GgufType::F32,
            dims: vec![spec.values as u64],
            payload: f32_bytes(&raw_values(&spec.role, spec.values)),
        });
    }
    let resid: Vec<f32> = (0..d.shape.hidden)
        .map(|i| ((i * 37) % 199) as f32 / 199.0 - 0.5)
        .collect();

    // The file on disk through the production reader.
    let dir = tempfile::tempdir().map_err(|e| EngError::Checkpoint(e.to_string()))?;
    let path = write_gguf(dir.path(), "layer.gguf", &entries);
    let limits = GgufLimits {
        max_header_bytes: 1 << 20,
        max_tensor_bytes: 1 << 28,
    };
    let weights = GgufWeights::open(&path, limits, None)?;

    // The manifests of the one- and two-card plans, narrowed to the file's
    // tensors exactly the way the session narrows `wanted` layers; the
    // placement decisions themselves are the manifest's, untouched.
    let table_one = ShardTable::new(&d.shape, 1, 32)?;
    let table = ShardTable::new(&d.shape, 2, 32)?;
    let file_names: std::collections::HashSet<String> =
        entries.iter().map(|entry| entry.name.clone()).collect();
    let narrow = |mut load: CardLoad| {
        load.entries
            .retain(|entry| file_names.contains(&entry.name));
        load
    };
    let mut one_loads = card_loads(&table_one, &inv, GgufType::Q4K)?;
    let mut loads = card_loads(&table, &inv, GgufType::Q4K)?;

    // The stack owns both card contexts; every load below runs on the
    // cards' own contexts through the borrowed loader, so no third
    // adapter lease exists at any point.
    let admission = served_admission();
    let (stack, mut states) = CardStack::new(&d, &options, &[0, 0], &admission)?;
    let load_options = LoadOptions {
        layout: WeightLayout::Native,
        ..LoadOptions::default()
    };
    let single_weights = load_card_weights(
        &BorrowedLoader::new(&stack.contexts()[0])?,
        &weights,
        &narrow(one_loads.remove(0)),
        0,
        load_options,
    )?;
    let card_weights = [
        load_card_weights(
            &BorrowedLoader::new(&stack.contexts()[0])?,
            &weights,
            &narrow(loads.remove(0)),
            0,
            load_options,
        )?,
        load_card_weights(
            &BorrowedLoader::new(&stack.contexts()[1])?,
            &weights,
            &narrow(loads.remove(0)),
            1,
            load_options,
        )?,
    ];

    // A CONSISTENT file whose attn_q declares half the rows loads fine but
    // must refuse at the production resolution: the card-local shape check
    // names the tensor. This is the wrongly-sliced-tensor gate.
    {
        let mut wrong = entries.clone();
        let q = wrong
            .iter_mut()
            .find(|entry| entry.name.contains("attn_q"))
            .expect("attn_q in the fixture");
        let per_row = (q.dims[0] as usize / 256) * 144;
        q.dims[1] /= 2;
        q.payload.truncate(per_row * q.dims[1] as usize);
        let wrong_path = write_gguf(dir.path(), "wrong-shape.gguf", &wrong);
        let wrong_weights = GgufWeights::open(&wrong_path, limits, None)?;
        let wrong_loaded = load_card_weights(
            &BorrowedLoader::new(&stack.contexts()[0])?,
            &wrong_weights,
            &narrow(card_loads(&table_one, &inv, GgufType::Q4K)?.remove(0)),
            0,
            load_options,
        )?;
        let err = match MixerWeights::new(
            &geometry,
            &table_layers,
            &wrong_loaded,
            &[layer],
            WeightLayout::Native,
        ) {
            Err(err) => err,
            Ok(_) => panic!("the half-rows attn_q must refuse at resolution"),
        };
        assert!(
            format!("{err:?}").contains("attn_q"),
            "the refusal names the tensor: {err:?}"
        );
    }

    // ---- From here the rig of run_rig, with the weights replaced by the
    // production resolution over the loaded sets.
    let ctx0 = &stack.contexts()[0];
    let stream0 = &stack.streams()[0];
    let scratch = KernelScratch::for_dims(&d, 1, options.kv_mode)?;
    let plan_ref = StepPlan::new(&d, 1, &options, &scratch)?;
    let arena_ref = ctx0.arena(&plan_ref.arena_tensors())?;
    let mut state_ref = ResidentState::allocate(ctx0, stream0, &d, &options)?;
    state_ref.write_step_params(stream0, &StepParams::new(0, 0)?)?;
    let write_row =
        |stream: &crate::kernels::cuda::DeviceStream,
         view: &crate::kernels::cuda::DeviceRange<'_>,
         bytes: &[u8]|
         -> Result<(), EngError> { view.buffer().write_at(stream, view.offset(), bytes) };
    write_row(
        stream0,
        &arena_ref.tensor(
            plan_ref
                .index(resid_slot)
                .ok_or_else(|| EngError::Checkpoint("resid".into()))?,
        )?,
        &f32_bytes(&resid),
    )?;
    for slot in [Slot::Q8Mix, Slot::Q8DeltaOut, Slot::Q8AttnOut] {
        if let Some(at) = plan_ref.index(slot) {
            let view = arena_ref.tensor(at)?;
            write_row(stream0, &view, &vec![0u8; view.bytes()])?;
        }
    }
    let mixers_ref = Mixers::new(
        ctx0,
        MixerParts {
            geometry: MixerGeometry::new(&d)?,
            table: &table_layers,
            weights: MixerWeights::new(
                &geometry,
                &table_layers,
                &single_weights,
                &[layer],
                WeightLayout::Native,
            )?,
            views: MixerViews::new(
                &plan_ref,
                &arena_ref,
                &geometry,
                WeightLayout::Native,
                options.kv_mode,
            )?,
            layout: WeightLayout::Native,
            kv_mode: options.kv_mode,
            paged: None,
            sparse: None,
        },
    )?;
    let resid_view_ref = arena_ref.tensor(
        plan_ref
            .index(resid_slot)
            .ok_or_else(|| EngError::Checkpoint("resid".into()))?,
    )?;
    mixers_ref.run_block(
        stream0,
        &state_ref,
        layer,
        BlockIo {
            resid: resid_view_ref,
            admit_pos: 0,
        },
    )?;
    stream0.synchronize()?;
    let read_slot = |slot: Slot| -> Result<Vec<f32>, EngError> {
        let view = arena_ref.tensor(
            plan_ref
                .index(slot)
                .ok_or_else(|| EngError::Checkpoint(format!("no {slot:?}")))?,
        )?;
        let mut bytes = vec![0u8; view.bytes()];
        view.buffer().read_at(stream0, view.offset(), &mut bytes)?;
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect())
    };
    let read_plane = |plane: Plane| -> Result<Vec<u8>, EngError> {
        let view = state_ref.kv_plane(ordinal, plane)?;
        let mut bytes = vec![0u8; view.bytes()];
        view.buffer().read_at(stream0, view.offset(), &mut bytes)?;
        Ok(bytes)
    };
    let mix_out_ref = read_slot(Slot::MixOut)?;
    let resid_out_ref = read_slot(resid_slot)?;
    let core_ref = read_slot(Slot::AttnCore)?;
    let planes_ref = (
        read_plane(Plane::K)?,
        read_plane(Plane::KScales)?,
        read_plane(Plane::V)?,
        read_plane(Plane::VScales)?,
    );

    // ---- The distributed block over the loaded per-card weight sets.
    for (card, state) in states.iter_mut().enumerate() {
        state.write_step_params(&stack.streams()[card], &StepParams::new(0, 0)?)?;
        let stream = &stack.streams()[card];
        let resid_at = stack.plans()[card]
            .index(resid_slot)
            .ok_or_else(|| EngError::Checkpoint("resid".into()))?;
        write_row(
            stream,
            &stack.shards()[card].tensor(resid_at)?,
            &f32_bytes(&resid),
        )?;
        for slot in [Slot::Q8Mix, Slot::Q8AttnOut] {
            if let Some(at) = stack.plans()[card].index(slot) {
                let view = stack.shards()[card].tensor(at)?;
                write_row(stream, &view, &vec![0u8; view.bytes()])?;
            }
        }
    }
    let kv_runs = [table.kv_rows(0)?, table.kv_rows(1)?];
    let head_runs = [table.query_heads(0)?, table.query_heads(1)?];
    let mut mixers: Vec<Mixers<'_>> = Vec::with_capacity(2);
    let mut ops_attn: Option<Vec<Vec<CardOp>>> = None;
    for (card, card_weight) in card_weights.iter().enumerate() {
        let g = MixerGeometry::for_card(&d, &table, card)?;
        let weights = MixerWeights::new(
            &g,
            &table_layers,
            card_weight,
            &[layer],
            WeightLayout::Native,
        )?;
        let ops = CardKindOps::new(&table, card, &g, options.kv_mode)?;
        let views = MixerViews::for_card(&stack.plans()[card], stack.shards()[card].arena(), &ops)?;
        ops_attn.get_or_insert_with(Vec::new).push(ops.attn.clone());
        let parts = MixerParts {
            geometry: g,
            table: &table_layers,
            weights,
            views,
            layout: WeightLayout::Native,
            kv_mode: options.kv_mode,
            paged: None,
            sparse: None,
        };
        mixers.push(Mixers::new(&stack.contexts()[card], parts)?);
    }
    let resid_views: Vec<crate::kernels::cuda::DeviceRange<'_>> = (0..2)
        .map(|card| {
            let at = stack.plans()[card]
                .index(resid_slot)
                .ok_or_else(|| EngError::Checkpoint("resid".into()))?;
            stack.shards()[card].tensor(at)
        })
        .collect::<Result<_, EngError>>()?;
    let ops = ops_attn
        .as_ref()
        .ok_or_else(|| EngError::Checkpoint("ops".into()))?;
    let split = ops[0]
        .iter()
        .position(|op| matches!(op, super::mixer_ops::CardOp::GatherV))
        .ok_or_else(|| EngError::Checkpoint("no gather".into()))?
        + 1;
    let ops_gather: Vec<&[CardOp]> = ops.iter().map(|l| &l[..split]).collect();
    run_card_block(
        &stack,
        &states,
        &mixers,
        layer,
        ordinal,
        &ops_gather,
        &resid_views,
        0,
        &kv_runs,
    )?;
    let rows_after_gather = (0..2)
        .map(|card| {
            stack.streams()[card].synchronize()?;
            let read = |slot: Slot| -> Result<Vec<f32>, EngError> {
                let view = mixers[card].slot_view(slot)?;
                let mut bytes = vec![0u8; view.bytes()];
                view.buffer()
                    .read_at(&stack.streams()[card], view.offset(), &mut bytes)?;
                Ok(bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes(*c))
                    .collect())
            };
            Ok((read(Slot::KRows)?, read(Slot::VRows)?))
        })
        .collect::<Result<Vec<_>, EngError>>()?;
    let ops_suffix: Vec<&[CardOp]> = ops.iter().map(|l| &l[split..]).collect();
    run_card_block(
        &stack,
        &states,
        &mixers,
        layer,
        ordinal,
        &ops_suffix,
        &resid_views,
        0,
        &kv_runs,
    )?;
    for card in 0..2 {
        stack.streams()[card].synchronize()?;
    }
    let read_card_slot = |card: usize, slot: Slot| -> Result<Vec<f32>, EngError> {
        let view = mixers[card].slot_view(slot)?;
        let mut bytes = vec![0u8; view.bytes()];
        view.buffer()
            .read_at(&stack.streams()[card], view.offset(), &mut bytes)?;
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect())
    };
    let read_card_plane = |card: usize, plane: Plane| -> Result<Vec<u8>, EngError> {
        let view = states[card].kv_plane(ordinal, plane)?;
        let mut bytes = vec![0u8; view.bytes()];
        view.buffer()
            .read_at(&stack.streams()[card], view.offset(), &mut bytes)?;
        Ok(bytes)
    };
    let mut words = Vec::with_capacity(2);
    for card in 0..2 {
        let mut word = [0u8; 4];
        stack.shards()[card]
            .ring_word()
            .read_at(&stack.streams()[card], 0, &mut word)?;
        words.push(u32::from_le_bytes(word));
    }
    let mix_out = (0..2)
        .map(|c| read_card_slot(c, Slot::MixOut))
        .collect::<Result<_, _>>()?;
    let resid_out = (0..2)
        .map(|c| {
            let at = stack.plans()[c]
                .index(resid_slot)
                .ok_or_else(|| EngError::Checkpoint("resid".into()))?;
            let view = stack.shards()[c].tensor(at)?;
            let mut bytes = vec![0u8; view.bytes()];
            view.buffer()
                .read_at(&stack.streams()[c], view.offset(), &mut bytes)?;
            Ok(bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect())
        })
        .collect::<Result<Vec<_>, EngError>>()?;
    let core = (0..2)
        .map(|c| read_card_slot(c, Slot::AttnCore))
        .collect::<Result<_, _>>()?;
    let mut planes = Vec::with_capacity(2);
    for card in 0..2 {
        planes.push((
            read_card_plane(card, Plane::K)?,
            read_card_plane(card, Plane::KScales)?,
            read_card_plane(card, Plane::V)?,
            read_card_plane(card, Plane::VScales)?,
        ));
    }
    Ok(Evidence {
        name,
        mix_out_ref,
        resid_out_ref,
        core_ref,
        planes_ref,
        rows_after_gather,
        mix_out,
        resid_out,
        core,
        planes,
        head_runs: head_runs.to_vec(),
        words,
    })
}

#[test]
fn a_real_gguf_file_loads_the_distributed_block() {
    const NAME: &str = "a_real_gguf_file_loads_the_distributed_block";
    if !device_present(NAME) {
        return;
    }
    let ev = run_gguf_rig(NAME).expect("the real-file rig");
    // Exact: the KV cache planes over the card's K/V head slice (the card's
    // planes are sliced to its heads; the tail is alignment padding).
    let full_heads = ev.head_runs[0].len() + ev.head_runs[1].len();
    for card in 0..2 {
        let got = &ev.planes[card];
        let want = &ev.planes_ref;
        for (plane, got_bytes, want_bytes) in [
            ("K codes", &got.0, &want.0),
            ("K scales", &got.1, &want.1),
            ("V codes", &got.2, &want.2),
            ("V scales", &got.3, &want.3),
        ] {
            let per_head = want_bytes.len() / full_heads;
            let start = ev.head_runs[card].start * per_head;
            let end = start + per_head * ev.head_runs[card].len();
            assert!(
                &got_bytes[..end - start] == &want_bytes[start..end],
                "{NAME}: card {card} {plane}: differing bytes over the card's head slice"
            );
        }
    }
    // Exact: the gathered rows agree across cards.
    assert_eq!(
        ev.rows_after_gather[0].0, ev.rows_after_gather[1].0,
        "the gathered K row is bit-identical"
    );
    assert_eq!(
        ev.rows_after_gather[0].1, ev.rows_after_gather[1].1,
        "the gathered V row is bit-identical"
    );
    // Exact: the per-head decode against the reference's head slice.
    let head_stride = ev.core_ref.len() / (ev.head_runs[0].len() + ev.head_runs[1].len());
    for card in 0..2 {
        let core = &ev.core[card];
        let start = ev.head_runs[card].start * head_stride;
        let slice = &ev.core_ref[start..start + core.len()];
        assert_eq!(
            max_abs_diff(core, slice),
            0.0,
            "{NAME}: card {card}: the per-head decode is expected exact"
        );
    }
    // Numerical: the summed mixer output and the residual epilogue.
    for card in 0..2 {
        let diff = max_abs_diff(&ev.mix_out[card], &ev.mix_out_ref);
        let bound = SPLIT_SUM_RELATIVE * max_abs(&ev.mix_out_ref).max(1.0);
        println!("{NAME}: card {card} MixOut max |diff| = {diff:.3e} (bound {bound:.3e})");
        assert!(
            diff <= bound,
            "{NAME}: card {card}: MixOut {diff} > {bound}"
        );
        let diff = max_abs_diff(&ev.resid_out[card], &ev.resid_out_ref);
        let bound = SPLIT_SUM_RELATIVE * max_abs(&ev.resid_out_ref).max(1.0);
        println!("{NAME}: card {card} resid max |diff| = {diff:.3e} (bound {bound:.3e})");
        assert!(diff <= bound, "{NAME}: card {card}: resid {diff} > {bound}");
    }
    for (card, word) in ev.words.iter().enumerate() {
        assert_eq!(*word, 0, "{NAME}: card {card}: ring clean");
    }
}
