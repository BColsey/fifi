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
    planes_ref: (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>),
    rows_after_gather: Vec<(Vec<f32>, Vec<f32>)>,
    mix_out: Vec<Vec<f32>>,
    resid_out: Vec<Vec<f32>>,
    core: Vec<Vec<f32>>,
    planes: Vec<(Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>)>,
    head_runs: Vec<std::ops::Range<usize>>,
    words: Vec<u32>,
}

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
    let mut stack = CardStack::new(&d, &options, &[0, 0], &admission)?;
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
    for card in 0..2 {
        stack.write_step_params(card, &StepParams::new(0, 0)?)?;
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
        let view = stack.shards()[card].state().kv_plane(ordinal, plane)?;
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
    // Exact: the KV cache planes (codes and scales) of the layer.
    for card in 0..2 {
        let got = &ev.planes[card];
        let want = &ev.planes_ref;
        for (plane, got_bytes, want_bytes) in [
            ("K codes", &got.0, &want.0),
            ("K scales", &got.1, &want.1),
            ("V codes", &got.2, &want.2),
            ("V scales", &got.3, &want.3),
        ] {
            let ranges = || {
                let mut out = Vec::new();
                let mut open: Option<usize> = None;
                for (i, (g, w)) in got_bytes.iter().zip(want_bytes).enumerate() {
                    if g != w {
                        open.get_or_insert(i);
                    } else if let Some(start) = open.take() {
                        out.push(format!("{start}..{i}"));
                    }
                }
                if let Some(start) = open {
                    out.push(format!("{start}..{}", got_bytes.len()));
                }
                out
            };
            let diff = ranges();
            println!(
                "{NAME}: card {card} {plane}: {} differing bytes, ranges {:?} of {}",
                got_bytes
                    .iter()
                    .zip(want_bytes)
                    .filter(|(g, w)| g != w)
                    .count(),
                &diff[..diff.len().min(8)],
                got_bytes.len()
            );
            assert!(
                diff.is_empty(),
                "card {card} {plane}: the appended cache must equal the reference exactly"
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
