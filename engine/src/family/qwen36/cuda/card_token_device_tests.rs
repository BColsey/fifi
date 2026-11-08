//! The whole-token rig: [`run_card_token`] - the eager two-card driver -
//! against the single-device reference over a REAL two-layer spec
//! (`[DeltaNet, Full]`, the model's head pattern), three chained GREEDY
//! tokens: token 0 embeds a fixed id, and every later token embeds the
//! merged argmax of the step before, both sides fed the SAME id.
//!
//! The reference is the production one-device path end to end on card 0's
//! context (`Mixers::run_block` per layer, `run_moe_block` per layer, the
//! final norm, the full-vocabulary head, the full-row argmax, all through
//! the resident launchers), over the same Q4_K sources the cards slice.
//!
//! Stage evidence:
//! * the greedy id: the HOST merge of the cards' shard argmaxes equals the
//!   reference argmax at EVERY token - the headline claim of the driver;
//! * each card's on-device argmax pair equals the host argmax over its own
//!   read-back logits shard (lowest id on a tie, NaN never wins) and its
//!   status word stays 0, as does the reference's;
//! * every layer boundary: each card's `resid_out(l)` row equals the
//!   reference's within the split-sum envelope (the partial matvecs differ
//!   by reduction order only), token after token, so no slot is stale or
//!   accumulating;
//! * the logits shards equal the reference row's ranges within the same
//!   envelope;
//! * the Full layer's KV planes stay BYTE-EXACT against the reference
//!   after the chain (the exactness proof of the block rig, held over the
//!   whole token);
//! * both ring words read 0 - no exchange failed.
use super::card_step::CardStack;
use super::card_token::{merge_head_candidates, run_card_token};
use super::dims::{KvMode, StepDims, StepOptions};
use super::layer_table::LayerTable;
use super::mixer::{BlockIo, CardKindOps, MixerParts, MixerViews, Mixers};
use super::mixer_geometry::MixerGeometry;
use super::mixer_ops::CardOp;
use super::mixer_weights::{DenseWeight, LayerWeights, MixerWeights};
use super::moe::{
    ExpertBufs, Launch, MoeBuffers, MoeDims, MoeLayerWeights, Packs, RouteBufs, StackWeights,
    run_moe_block,
};
use super::native_device_rig::{Rng, device_present, q4k_rows, raw_values};
use super::shard::ShardTable;
use super::state::ResidentState;
use super::state::StepField;
use super::state_layout::{Plane, Word};
use super::step_ops::{final_resid, resid_in, resid_out};
use super::step_plan::{KernelScratch, Slot, StepPlan};
use crate::EngError;
use crate::family::gguf_model::Role;
use crate::family::project::cuda::WeightLayout;
use crate::family::project::cuda::served_admission;
use crate::family::qwen36::host::HostShape;
use crate::family::{AttentionKind, FamilySpec, QWEN3_6_35B_A3B, gguf_model::Inventory};
use crate::kernels::cuda::plan::{assemble_slice, plan_slice};
use crate::kernels::cuda::resident::add_rmsnorm_q8::AddNormQ8Call;
use crate::kernels::cuda::resident::embed::EmbedNativeCall;
use crate::kernels::cuda::resident::matvec_q::MatvecCall;
use crate::kernels::cuda::resident::step_params::{ArgmaxCall, StepParams};
use crate::kernels::cuda::{CudaContext, DeviceBuffer, NativeWeight, QuantTensor};
use crate::weights::GgufType;

const F32: usize = size_of::<f32>();
const Q4K_BLOCK: usize = 144;
const Q4K_VALUES: usize = 256;
/// The split-sum envelope: the per-card partial matvecs differ from the
/// fused one by reduction order only.
const SPLIT_SUM_RELATIVE: f32 = 1e-3;

/// The DeltaNet state gate. A value that crosses one q8 bin boundary of
/// the normalized input row (~0.8% of the row's range) shifts its whole
/// head block, so the recurrent and conv state carry a few per-mille of
/// relative drift once the chain feeds residual envelope drift back in.
/// Bit equality at identical inputs is the block rig's own gate.
const STATE_ENVELOPE_RELATIVE: f32 = 5e-3;
const TOKENS: usize = 3;
/// The seed token of the greedy chain.
const FIRST_TOKEN: u32 = 7;

static TWO_LAYERS: [AttentionKind; 2] = [AttentionKind::DeltaNet, AttentionKind::Full];

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

/// The real 40-layer model spec with ONE memory-bound reduction: 32
/// experts instead of 256. The full 256-expert stacks are ~18 GB of Q4_K
/// per side - ~36 GB for the reference plus the two card shards on one
/// 3090; 32 experts keep the per-expert-distinct-weights structure at
/// ~4.5 GB. Everything else is the shipped shape: 40 layers, the real
/// kinds pattern, hidden 2048, inter 512, top_k 8, vocab 152064.
fn real_layer_spec() -> FamilySpec {
    FamilySpec {
        experts: 32,
        ..QWEN3_6_35B_A3B
    }
}

/// Best logit value and the best among the rest, over shard slices. The
/// gap between them says how well separated the greedy winner is.
fn top2_of<'a>(rows: impl IntoIterator<Item = &'a [f32]>) -> (f32, f32) {
    let (mut best, mut runner) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for row in rows {
        for &v in row {
            if v > best {
                runner = best;
                best = v;
            } else if v > runner {
                runner = v;
            }
        }
    }
    (best, runner)
}

/// One KV plane's byte divergence against the reference. A card's KV row
/// is written from its own projection of a residual that already sits
/// inside the split-sum envelope, so an int8 code or an fp16 scale can
/// flip at a bin boundary: a 1-LSB flip on at most one percent of a
/// plane's bytes is the accepted drift class.
#[derive(Clone, Copy, Debug)]
struct PlaneDrift {
    diffs: usize,
    bytes: usize,
    max_delta: u8,
}

impl PlaneDrift {
    fn of(want: &[u8], got: &[u8]) -> Self {
        let diffs = want.iter().zip(got).filter(|(a, b)| a != b).count();
        let max_delta = want
            .iter()
            .zip(got)
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap_or(0);
        Self {
            diffs,
            bytes: want.len(),
            max_delta,
        }
    }

    fn within_bound(&self) -> bool {
        self.max_delta <= 1 && self.diffs * 100 <= self.bytes
    }
}

/// The rig's own spec: the real 35B geometry with TWO layers - DeltaNet
/// then Full, the model's head pattern - and a vocabulary the rig can
/// afford twice (embedding table plus head).
fn rig_spec() -> FamilySpec {
    FamilySpec {
        layers: 2,
        vocab: 4096,
        layers_attention: &TWO_LAYERS,
        ..QWEN3_6_35B_A3B
    }
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn bytes_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
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

/// The host argmax the native kernel pins: largest value, lowest index on
/// a tie, a NaN never wins, -1 when nothing is comparable.
fn host_argmax(values: &[f32]) -> i32 {
    let mut best: Option<(usize, f32)> = None;
    for (i, &value) in values.iter().enumerate() {
        if value.is_nan() {
            continue;
        }
        let better = match best {
            Some((_, top)) => value > top,
            None => true,
        };
        if better {
            best = Some((i, value));
        }
    }
    best.map(|(i, _)| i as i32).unwrap_or(-1)
}

fn within_envelope(name: &str, card: usize, got: &[f32], want: &[f32]) {
    let diff = max_abs_diff(got, want);
    let bound = max_abs(want) * SPLIT_SUM_RELATIVE + 1e-6;
    assert!(
        diff <= bound,
        "{name} card {card}: max diff {diff:.3e} over bound {bound:.3e} (row max {})",
        max_abs(want)
    );
}

fn within_state_envelope(name: &str, card: usize, got: &[f32], want: &[f32]) {
    let diff = max_abs_diff(got, want);
    let bound = max_abs(want) * STATE_ENVELOPE_RELATIVE + 1e-5;
    assert!(
        diff <= bound,
        "{name} card {card}: max diff {diff:.3e} over bound {bound:.3e} (row max {})",
        max_abs(want)
    );
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

fn row_slice(source: &[u8], k: usize, rows: std::ops::Range<usize>) -> Vec<u8> {
    let per_row = k / Q4K_VALUES * Q4K_BLOCK;
    source[rows.start * per_row..rows.end * per_row].to_vec()
}

/// Rows `range` of every expert's block of a Q4_K expert stack.
fn expert_row_slice(
    source: &[u8],
    experts: usize,
    rows_per: usize,
    k: usize,
    range: std::ops::Range<usize>,
) -> Vec<u8> {
    let per_row = k / Q4K_VALUES * Q4K_BLOCK;
    let mut out = Vec::with_capacity(experts * range.len() * per_row);
    for expert in 0..experts {
        let at = (expert * rows_per + range.start) * per_row;
        out.extend_from_slice(&source[at..at + range.len() * per_row]);
    }
    out
}

/// Columns `range` of every row of every expert's block of a Q4_K stack.
fn expert_col_slice(
    source: &[u8],
    experts: usize,
    rows_per: usize,
    k: usize,
    range: std::ops::Range<usize>,
) -> Vec<u8> {
    let blocks = k / Q4K_VALUES;
    let take = (range.end - range.start) / Q4K_VALUES;
    let mut out = Vec::with_capacity(experts * rows_per * take * Q4K_BLOCK);
    for row in 0..experts * rows_per {
        let at = (row * blocks + range.start / Q4K_VALUES) * Q4K_BLOCK;
        out.extend_from_slice(&source[at..at + take * Q4K_BLOCK]);
    }
    out
}

/// The MoE fixture of one layer, at the full dims.
struct MoeSources {
    gate: Vec<u8>,
    up: Vec<u8>,
    down: Vec<u8>,
    shared_gate: Vec<u8>,
    shared_up: Vec<u8>,
    shared_down: Vec<u8>,
    router: Vec<f32>,
    router_shared: Vec<f32>,
}

fn moe_sources(dims: &MoeDims, seed: u64) -> MoeSources {
    let mut rng = Rng::new(seed);
    let expert_rows =
        |rng: &mut Rng, rows_per: usize, k: usize| q4k_rows(dims.n_expert * rows_per, k, rng);
    let row = |rng: &mut Rng, n: usize| {
        (0..n)
            .map(|_| (rng.below(200) as f32 - 100.0) / 400.0)
            .collect()
    };
    MoeSources {
        gate: expert_rows(&mut rng, dims.inter, dims.hidden),
        up: expert_rows(&mut rng, dims.inter, dims.hidden),
        down: expert_rows(&mut rng, dims.hidden, dims.inter),
        shared_gate: q4k_rows(dims.shared_inter, dims.hidden, &mut rng),
        shared_up: q4k_rows(dims.shared_inter, dims.hidden, &mut rng),
        shared_down: q4k_rows(dims.hidden, dims.shared_inter, &mut rng),
        router: row(&mut rng, dims.n_expert * dims.hidden),
        router_shared: row(&mut rng, dims.hidden),
    }
}

/// The owned MoE weights of one side at its own width; the raw router rows
/// are borrowed from the reference (only card 0 reads them).
struct MoeOwned {
    stacks: Vec<NativeWeight>,
    denses: Vec<NativeWeight>,
}

impl MoeOwned {
    /// The expert stacks carry their per-expert block shape, exactly as the
    /// loader does for a 3-D GGUF tensor.
    fn stack(
        ctx: &CudaContext,
        source: &[u8],
        rows: usize,
        k: usize,
        experts: usize,
    ) -> Result<NativeWeight, EngError> {
        Ok(upload_native(ctx, source, rows, k)?.with_experts(Some(experts)))
    }

    fn full(ctx: &CudaContext, s: &MoeSources, dims: &MoeDims) -> Result<Self, EngError> {
        Ok(Self {
            stacks: vec![
                Self::stack(
                    ctx,
                    &s.gate,
                    dims.n_expert * dims.inter,
                    dims.hidden,
                    dims.n_expert,
                )?,
                Self::stack(
                    ctx,
                    &s.up,
                    dims.n_expert * dims.inter,
                    dims.hidden,
                    dims.n_expert,
                )?,
                Self::stack(
                    ctx,
                    &s.down,
                    dims.n_expert * dims.hidden,
                    dims.inter,
                    dims.n_expert,
                )?,
            ],
            denses: vec![
                upload_native(ctx, &s.shared_gate, dims.shared_inter, dims.hidden)?,
                upload_native(ctx, &s.shared_up, dims.shared_inter, dims.hidden)?,
                upload_native(ctx, &s.shared_down, dims.hidden, dims.shared_inter)?,
            ],
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn card(
        ctx: &CudaContext,
        s: &MoeSources,
        full: &MoeDims,
        dims: &MoeDims,
        inter: &std::ops::Range<usize>,
        shared_inter: &std::ops::Range<usize>,
    ) -> Result<Self, EngError> {
        let gate = expert_row_slice(
            &s.gate,
            full.n_expert,
            full.inter,
            full.hidden,
            inter.clone(),
        );
        let up = expert_row_slice(&s.up, full.n_expert, full.inter, full.hidden, inter.clone());
        let down = expert_col_slice(
            &s.down,
            full.n_expert,
            full.hidden,
            full.inter,
            inter.clone(),
        );
        let sg = expert_row_slice(
            &s.shared_gate,
            1,
            full.shared_inter,
            full.hidden,
            shared_inter.clone(),
        );
        let su = expert_row_slice(
            &s.shared_up,
            1,
            full.shared_inter,
            full.hidden,
            shared_inter.clone(),
        );
        let sd = expert_col_slice(
            &s.shared_down,
            1,
            full.hidden,
            full.shared_inter,
            shared_inter.clone(),
        );
        Ok(Self {
            stacks: vec![
                Self::stack(
                    ctx,
                    &gate,
                    full.n_expert * dims.inter,
                    full.hidden,
                    full.n_expert,
                )?,
                Self::stack(
                    ctx,
                    &up,
                    full.n_expert * dims.inter,
                    full.hidden,
                    full.n_expert,
                )?,
                Self::stack(
                    ctx,
                    &down,
                    full.n_expert * full.hidden,
                    dims.inter,
                    full.n_expert,
                )?,
            ],
            denses: vec![
                upload_native(ctx, &sg, dims.shared_inter, full.hidden)?,
                upload_native(ctx, &su, dims.shared_inter, full.hidden)?,
                upload_native(ctx, &sd, full.hidden, dims.shared_inter)?,
            ],
        })
    }

    fn moe<'a>(
        &'a self,
        router: &'a DeviceBuffer,
        router_shared: &'a DeviceBuffer,
    ) -> MoeLayerWeights<'a> {
        MoeLayerWeights {
            router,
            router_shared,
            gate: StackWeights::Native(&self.stacks[0]),
            up: StackWeights::Native(&self.stacks[1]),
            down: StackWeights::Native(&self.stacks[2]),
            shared_gate: DenseWeight::Native(&self.denses[0]),
            shared_up: DenseWeight::Native(&self.denses[1]),
            shared_down: DenseWeight::Native(&self.denses[2]),
        }
    }
}

/// One greedy step's readings.
struct StepEvidence {
    ref_id: i32,
    ref_status: i32,
    merged_id: i32,
    /// Per card: (local pair token, local pair status, logits shard).
    cards: Vec<(i32, i32, Vec<f32>)>,
    /// Per card, per layer: the card's `resid_out(l)` row.
    boundaries: Vec<Vec<Vec<f32>>>,
    /// Per layer: the reference's `resid_out(l)` row.
    ref_boundaries: Vec<Vec<f32>>,
    ref_logits: Vec<f32>,
    /// Winner-minus-runner-up logit gap, from the reference row and from
    /// the merged shards.
    ref_margin: f32,
    merged_margin: f32,
}

struct Evidence {
    steps: Vec<StepEvidence>,
    ring_words: Vec<u32>,
    full_ordinals: Vec<usize>,
    /// [card][full ordinal][plane in K, KScales, V, VScales order].
    kv_drift: Vec<Vec<Vec<PlaneDrift>>>,
    /// The global row offset of each card's head shard.
    starts: Vec<usize>,
}

fn run_rig(spec: FamilySpec, tokens: usize, independent: bool) -> Result<Evidence, EngError> {
    let inv = Inventory::for_spec(&spec)?;
    let shape = HostShape::from_spec(&spec, 64)?;
    let d = StepDims::new(&shape, &inv)?;
    let options = options();
    let layers = shape.kinds.len();
    let table_layers = LayerTable::new(&d, &inv)?;
    let kinds: Vec<AttentionKind> = (0..layers)
        .map(|l| Ok(table_layers.slot(l)?.kind))
        .collect::<Result<_, EngError>>()?;
    let ordinals: Vec<usize> = (0..layers)
        .map(|l| Ok(table_layers.slot(l)?.ordinal))
        .collect::<Result<_, EngError>>()?;
    let full_ordinals: Vec<usize> = kinds
        .iter()
        .zip(ordinals.iter())
        .filter(|&(&k, _)| k == AttentionKind::Full)
        .map(|(_, &o)| o)
        .collect();
    let delta_ordinals: Vec<usize> = kinds
        .iter()
        .zip(ordinals.iter())
        .filter(|&(&k, _)| k == AttentionKind::DeltaNet)
        .map(|(_, &o)| o)
        .collect();

    let admission = served_admission();
    let (stack, mut states) = CardStack::new(&d, &options, &[0, 0], &admission)?;
    for (card, state) in states.iter_mut().enumerate() {
        state.write_step_params(
            &stack.streams()[card],
            &StepParams::new(FIRST_TOKEN as i32, 0)?,
        )?;
    }
    let ctx0 = &stack.contexts()[0];
    let stream0 = &stack.streams()[0];
    let hidden = d.shape.hidden;
    let vocab = shape.vocab;
    let eps = d.shape.eps;
    let table = ShardTable::new(&d.shape, 2, 32)?;

    // ---- The fixtures, one rng stream over the whole model.
    let mut rng = Rng::new(0x707_0E7);
    let geometry_full = MixerGeometry::new(&d)?;
    type DenseSource = Vec<(Role, Vec<u8>, usize, usize)>;
    type RawSource = Vec<(Role, Vec<f32>)>;
    let mut dense_sources: Vec<(AttentionKind, DenseSource)> = Vec::new();
    let mut raw_sources: Vec<(AttentionKind, RawSource)> = Vec::new();
    let unique_kinds: Vec<AttentionKind> = {
        let mut seen = Vec::new();
        for &kind in &kinds {
            if !seen.contains(&kind) {
                seen.push(kind);
            }
        }
        seen
    };
    for &kind in &unique_kinds {
        let specs = geometry_full.dense_specs(WeightLayout::Native, kind)?;
        let per_kind: Vec<(Role, Vec<u8>, usize, usize)> = specs
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
        dense_sources.push((kind, per_kind));
        let raws: Vec<(Role, Vec<f32>)> = geometry_full
            .raw_specs(kind)?
            .iter()
            .map(|spec| (spec.role, raw_values(&spec.role, spec.values)))
            .collect();
        raw_sources.push((kind, raws));
    }
    let moe_full_dims = MoeDims::new(&d, 1);
    let moe_fixtures: Vec<MoeSources> = (0..layers)
        .map(|l| moe_sources(&moe_full_dims, 0x50E_FEED + l as u64))
        .collect();
    let embed_source = q4k_rows(vocab, hidden, &mut rng);
    let head_source = q4k_rows(vocab, hidden, &mut rng);
    let output_norm: Vec<f32> = (0..hidden)
        .map(|_| (rng.below(200) as f32 - 100.0) / 400.0 + 0.75)
        .collect();
    let dense_source_for =
        |kind: AttentionKind, role: Role| -> Option<&(Role, Vec<u8>, usize, usize)> {
            dense_sources
                .iter()
                .flat_map(|(k, s)| s.iter().map(move |r| (*k, r)))
                .find(|(k, r)| *k == kind && r.0 == role)
                .map(|(_, r)| r)
        };

    // ---- The reference: the full production step on card 0's context.
    let scratch = KernelScratch::for_dims(&d, 1, options.kv_mode)?;
    let plan_ref = StepPlan::new(&d, 1, &options, &scratch)?;
    let arena_ref = ctx0.arena(&plan_ref.arena_tensors())?;
    let mut state_ref = ResidentState::allocate(ctx0, stream0, &d, &options)?;
    let embed_ref = upload_native(ctx0, &embed_source, vocab, hidden)?;
    let head_ref = upload_native(ctx0, &head_source, vocab, hidden)?;
    let norm_ref = ctx0.upload(&f32_bytes(&output_norm))?;
    let ref_view = |slot: Slot| -> Result<crate::kernels::cuda::DeviceRange<'_>, EngError> {
        let at = plan_ref
            .index(slot)
            .ok_or_else(|| EngError::Checkpoint(format!("ref: no {slot:?} slot")))?;
        arena_ref.tensor(at)
    };
    for slot in [
        Slot::Q8Mix,
        Slot::Q8DeltaOut,
        Slot::Q8AttnOut,
        Slot::Q8Final,
    ] {
        if plan_ref.index(slot).is_some() {
            let view = ref_view(slot)?;
            view.buffer()
                .write_at(stream0, view.offset(), &vec![0u8; view.bytes()])?;
        }
    }
    let mut ref_dense_sets: Vec<Vec<NativeWeight>> = Vec::with_capacity(layers);
    let mut ref_raw_sets: Vec<Vec<(Role, DeviceBuffer)>> = Vec::with_capacity(layers);
    for &kind in &kinds {
        let specs = geometry_full.dense_specs(WeightLayout::Native, kind)?;
        let natives: Vec<NativeWeight> = specs
            .iter()
            .map(|spec| {
                let (_, source, _, _) = dense_source_for(kind, spec.role)
                    .ok_or_else(|| EngError::Checkpoint(format!("ref: no {:?}", spec.role)))?;
                upload_native(ctx0, source, spec.rows, spec.k)
            })
            .collect::<Result<_, _>>()?;
        let raws = raw_sources
            .iter()
            .find(|(k, _)| *k == kind)
            .map(|(_, r)| r)
            .ok_or_else(|| EngError::Checkpoint("ref: no raws".into()))?;
        let raw_bufs: Vec<(Role, DeviceBuffer)> = raws
            .iter()
            .map(|(role, values)| Ok((*role, ctx0.upload(&f32_bytes(values))?)))
            .collect::<Result<_, EngError>>()?;
        ref_dense_sets.push(natives);
        ref_raw_sets.push(raw_bufs);
    }
    let mut entries: Vec<Option<LayerWeights<'_>>> = (0..layers).map(|_| None).collect();
    for (l, &kind) in kinds.iter().enumerate() {
        let specs = geometry_full.dense_specs(WeightLayout::Native, kind)?;
        let dense: Vec<(Role, DenseWeight<'_>)> = specs
            .iter()
            .zip(&ref_dense_sets[l])
            .map(|(spec, native)| (spec.role, DenseWeight::Native(native)))
            .collect();
        let raw_views: Vec<(Role, crate::kernels::cuda::DeviceRange<'_>)> = ref_raw_sets[l]
            .iter()
            .map(|(role, buf)| Ok((*role, buf.range(0, buf.bytes())?)))
            .collect::<Result<_, EngError>>()?;
        entries[l] = Some(LayerWeights::for_tests(dense, raw_views));
    }
    let parts_ref = MixerParts {
        geometry: geometry_full,
        table: &table_layers,
        weights: MixerWeights::from_layers(entries),
        views: MixerViews::new(
            &plan_ref,
            &arena_ref,
            &geometry_full,
            WeightLayout::Native,
            options.kv_mode,
        )?,
        layout: WeightLayout::Native,
        kv_mode: options.kv_mode,
        paged: None,
        sparse: None,
    };
    let mixers_ref = Mixers::new(ctx0, parts_ref)?;
    // The reference MoE weights and buffers.
    let routers: Vec<(DeviceBuffer, DeviceBuffer)> = moe_fixtures
        .iter()
        .map(|s| {
            Ok((
                ctx0.upload(&f32_bytes(&s.router))?,
                ctx0.upload(&f32_bytes(&s.router_shared))?,
            ))
        })
        .collect::<Result<_, EngError>>()?;
    let moe_ref_owned: Vec<MoeOwned> = moe_fixtures
        .iter()
        .map(|s| MoeOwned::full(ctx0, s, &moe_full_dims))
        .collect::<Result<_, _>>()?;
    let moe_ref: Vec<MoeLayerWeights<'_>> = moe_ref_owned
        .iter()
        .zip(&routers)
        .map(|(owned, (router, shared))| owned.moe(router, shared))
        .collect();
    let ref_moe_view =
        |_parity: usize, slot: Slot| -> Result<crate::kernels::cuda::DeviceRange<'_>, EngError> {
            let at = plan_ref
                .index(slot)
                .ok_or_else(|| EngError::Checkpoint(format!("ref moe: no {slot:?} slot")))?;
            arena_ref.tensor(at)
        };

    // ---- The cards: sliced weights, card op lists, card views.
    let kv_runs: Vec<std::ops::Range<usize>> = (0..2)
        .map(|card| table.kv_rows(card))
        .collect::<Result<_, _>>()?;
    let q_runs: Vec<std::ops::Range<usize>> = (0..2)
        .map(|card| table.attn_q_rows(card))
        .collect::<Result<_, _>>()?;
    let o_cols: Vec<std::ops::Range<usize>> = (0..2)
        .map(|card| table.attn_o_cols(card))
        .collect::<Result<_, _>>()?;
    let head_runs: Vec<std::ops::Range<usize>> = (0..2)
        .map(|card| table.head_rows(card))
        .collect::<Result<_, _>>()?;
    let inter: Vec<std::ops::Range<usize>> = (0..2)
        .map(|card| table.expert_inter(card))
        .collect::<Result<_, _>>()?;
    let shared_inter: Vec<std::ops::Range<usize>> = (0..2)
        .map(|card| table.shared_inter(card))
        .collect::<Result<_, _>>()?;
    let conv_segments: Vec<[std::ops::Range<usize>; 3]> = (0..2)
        .map(|card| table.delta_qkv_segments(card))
        .collect::<Result<_, _>>()?;
    // Per card: dense natives per layer, raw bufs per layer.
    let mut card_dense: Vec<Vec<Vec<NativeWeight>>> = Vec::with_capacity(2);
    let mut card_raws: Vec<Vec<Vec<(Role, DeviceBuffer)>>> = Vec::with_capacity(2);
    let mut card_kind_ops: Vec<CardKindOps> = Vec::with_capacity(2);
    let shape_delta = &d.shape.delta;
    for card in 0..2 {
        let ctx = &stack.contexts()[card];
        let g = MixerGeometry::for_card(&d, &table, card)?;
        let mut per_layer_natives = Vec::with_capacity(layers);
        let mut per_layer_raws = Vec::with_capacity(layers);
        for &kind in &kinds {
            let specs = g.dense_specs(WeightLayout::Native, kind)?;
            let mut natives = Vec::with_capacity(specs.len());
            for spec in &specs {
                let (_, source, rows, k) = dense_source_for(kind, spec.role)
                    .ok_or_else(|| EngError::Checkpoint(format!("card: no {:?}", spec.role)))?;
                let (rows, k) = (*rows, *k);
                let sliced: Vec<u8> = match (kind, spec.role) {
                    (AttentionKind::Full, Role::AttnQ) => {
                        row_slice(source, k, q_runs[card].clone())
                    }
                    (AttentionKind::Full, Role::AttnK | Role::AttnV) => {
                        row_slice(source, k, kv_runs[card].clone())
                    }
                    (AttentionKind::Full, Role::AttnOutput) => {
                        let slice = plan_slice(
                            "rig",
                            GgufType::Q4K,
                            &[k as u64, rows as u64],
                            o_cols[card].start as u64..o_cols[card].end as u64,
                            0..rows as u64,
                        )?;
                        assemble_slice(source, &slice)?
                    }
                    (AttentionKind::DeltaNet, Role::AttnQkv) => conv_segments[card]
                        .iter()
                        .flat_map(|seg| row_slice(source, k, seg.clone()))
                        .collect(),
                    (AttentionKind::DeltaNet, Role::AttnGate) => {
                        row_slice(source, k, table.delta_z_rows(card)?)
                    }
                    (AttentionKind::DeltaNet, Role::SsmBeta | Role::SsmAlpha) => {
                        row_slice(source, k, table.value_heads(card)?)
                    }
                    (AttentionKind::DeltaNet, Role::SsmOut) => {
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
                natives.push(upload_native(ctx, &sliced, spec.rows, spec.k)?);
            }
            per_layer_natives.push(natives);
            let raws = raw_sources
                .iter()
                .find(|(k, _)| *k == kind)
                .map(|(_, r)| r)
                .ok_or_else(|| EngError::Checkpoint("card: no raws".into()))?;
            let card_raw: Vec<(Role, DeviceBuffer)> = raws
                .iter()
                .map(|(role, values)| {
                    let sliced: Vec<f32> = match (kind, *role) {
                        (AttentionKind::DeltaNet, Role::SsmA | Role::SsmDtBias) => {
                            values[table.value_heads(card)?].to_vec()
                        }
                        (AttentionKind::DeltaNet, Role::SsmConv1d) => conv_segments[card]
                            .iter()
                            .flat_map(|seg| {
                                values[seg.start * shape_delta.conv_kernel
                                    ..seg.end * shape_delta.conv_kernel]
                                    .to_vec()
                            })
                            .collect(),
                        _ => values.clone(),
                    };
                    Ok((*role, ctx.upload(&f32_bytes(&sliced))?))
                })
                .collect::<Result<_, EngError>>()?;
            per_layer_raws.push(card_raw);
        }
        card_dense.push(per_layer_natives);
        card_raws.push(per_layer_raws);
        card_kind_ops.push(CardKindOps::new(&table, card, &g, options.kv_mode)?);
    }
    // Per card: ONE Mixers holding both layers' weights (`entries` is
    // per-layer), the same shape `run_card_block` reads layer `l` from.
    let mut mixers: Vec<Mixers<'_>> = Vec::with_capacity(2);
    let mut ops_owned: Vec<Vec<Vec<CardOp>>> = (0..layers).map(|_| Vec::new()).collect();
    for card in 0..2 {
        let g = MixerGeometry::for_card(&d, &table, card)?;
        let ops = &card_kind_ops[card];
        let mut entries: Vec<Option<LayerWeights<'_>>> = (0..layers).map(|_| None).collect();
        for (l, &kind) in kinds.iter().enumerate() {
            let specs = g.dense_specs(WeightLayout::Native, kind)?;
            let dense: Vec<(Role, DenseWeight<'_>)> = specs
                .iter()
                .zip(&card_dense[card][l])
                .map(|(spec, native)| (spec.role, DenseWeight::Native(native)))
                .collect();
            let raw_views: Vec<(Role, crate::kernels::cuda::DeviceRange<'_>)> = card_raws[card][l]
                .iter()
                .map(|(role, buf)| Ok((*role, buf.range(0, buf.bytes())?)))
                .collect::<Result<_, EngError>>()?;
            entries[l] = Some(LayerWeights::for_tests(dense, raw_views));
            ops_owned[l].push(match kind {
                AttentionKind::Full => ops.attn.clone(),
                AttentionKind::DeltaNet => ops.delta.clone(),
                other => {
                    return Err(EngError::Checkpoint(format!("no card ops for {other:?}")));
                }
            });
        }
        let parts = MixerParts {
            geometry: g,
            table: &table_layers,
            weights: MixerWeights::from_layers(entries),
            views: MixerViews::for_card(&stack.plans()[card], stack.shards()[card].arena(), ops)?,
            layout: WeightLayout::Native,
            kv_mode: options.kv_mode,
            paged: None,
            sparse: None,
        };
        mixers.push(Mixers::new(&stack.contexts()[card], parts)?);
    }
    // Per card: MoE weights per layer, the head shard, the final norm.
    let card_moe_dims: Vec<Vec<MoeDims>> = (0..2)
        .map(|card| {
            (0..layers)
                .map(|_| MoeDims::for_card(&d, &table, card, 1))
                .collect::<Result<_, _>>()
        })
        .collect::<Result<_, _>>()?;
    let card_moe_owned: Vec<Vec<MoeOwned>> = (0..2)
        .map(|card| {
            (0..layers)
                .map(|l| {
                    MoeOwned::card(
                        &stack.contexts()[card],
                        &moe_fixtures[l],
                        &moe_full_dims,
                        &card_moe_dims[card][l],
                        &inter[card],
                        &shared_inter[card],
                    )
                })
                .collect::<Result<_, _>>()
        })
        .collect::<Result<_, _>>()?;
    let moe_sets: Vec<Vec<MoeLayerWeights<'_>>> = (0..layers)
        .map(|l| {
            (0..2)
                .map(|card| card_moe_owned[card][l].moe(&routers[l].0, &routers[l].1))
                .collect()
        })
        .collect();
    let moe_refs: Vec<&[MoeLayerWeights<'_>]> = moe_sets.iter().map(|s| s.as_slice()).collect();
    let head_owners: Vec<NativeWeight> = (0..2)
        .map(|card| {
            upload_native(
                &stack.contexts()[card],
                &row_slice(&head_source, hidden, head_runs[card].clone()),
                head_runs[card].len(),
                hidden,
            )
        })
        .collect::<Result<_, _>>()?;
    let heads: Vec<&NativeWeight> = head_owners.iter().collect();
    let output_norm_owners: Vec<DeviceBuffer> = (0..2)
        .map(|card| stack.contexts()[card].upload(&f32_bytes(&output_norm)))
        .collect::<Result<_, _>>()?;
    let output_norms: Vec<&DeviceBuffer> = output_norm_owners.iter().collect();
    let block_ops: Vec<Vec<&[CardOp]>> = ops_owned
        .iter()
        .map(|per_layer| per_layer.iter().map(|l| l.as_slice()).collect::<Vec<_>>())
        .collect();
    let block_ops_refs: Vec<&[&[CardOp]]> = block_ops.iter().map(|v| v.as_slice()).collect();
    // Zero the card packs once, as the reference does.
    for card in 0..2 {
        let stream = &stack.streams()[card];
        for slot in [
            Slot::Q8Mix,
            Slot::Q8DeltaOut,
            Slot::Q8AttnOut,
            Slot::Q8Final,
        ] {
            if let Some(at) = stack.plans()[card].index(slot) {
                let view = stack.shards()[card].tensor(at)?;
                view.buffer()
                    .write_at(stream, view.offset(), &vec![0u8; view.bytes()])?;
            }
        }
    }

    let card_view =
        |card: usize, slot: Slot| -> Result<crate::kernels::cuda::DeviceRange<'_>, EngError> {
            let at = stack.plans()[card]
                .index(slot)
                .ok_or_else(|| EngError::Checkpoint(format!("card {card}: no {slot:?} slot")))?;
            stack.shards()[card].tensor(at)
        };

    // ---- The greedy chain.
    let mut steps = Vec::with_capacity(tokens);
    // Teacher-forced mode feeds both sides the merged id; independent
    // mode feeds the reference its OWN id, so sequence equality proves
    // the coordinator's real feedback path.
    let mut ref_token = FIRST_TOKEN;
    let mut card_token;
    for t in 0..tokens {
        // The reference step first. The local borrows keep the fn-scope
        // ones dead here, so the driver's `&mut stack` below is free.
        let (ref_id, ref_status, ref_logits, ref_boundaries) = {
            let ctx0 = &stack.contexts()[0];
            let stream0 = &stack.streams()[0];
            // The production order: the start block, then the row admit.
            state_ref.write_step_params(stream0, &StepParams::new(ref_token as i32, t as i32)?)?;
            let admitted = state_ref.enqueue_rows(1)?;
            let params = state_ref.word(Word::StepParams)?;
            ctx0.step_begin(stream0, params)?;
            let ids = state_ref.step_field(StepField::TokenId)?;
            ctx0.embed_native(
                stream0,
                EmbedNativeCall {
                    weight: &embed_ref,
                    ids,
                    x: ref_view(Slot::ResidA)?,
                    status: state_ref.step_status()?,
                    m: 1,
                },
            )?;
            for (l, moe) in moe_ref.iter().enumerate() {
                mixers_ref.run_block(
                    stream0,
                    &state_ref,
                    l,
                    BlockIo {
                        resid: ref_view(resid_in(l))?,
                        admit_pos: admitted.admit_pos,
                    },
                )?;
                let ref_moe_set = MoeBuffers {
                    route: RouteBufs {
                        resid_in: ref_view(resid_in(l))?,
                        resid_out: ref_view(resid_out(l))?,
                        norm_row: ref_view(Slot::NormRow)?,
                        logits: ref_view(Slot::RouterLogits)?,
                        ids: ref_view(Slot::TopkId)?,
                        weights: ref_view(Slot::TopkWeight)?,
                        shared_gate: ref_view(Slot::SharedGate)?,
                        status: state_ref.step_status()?,
                    },
                    expert: ExpertBufs {
                        packs: Packs::Q8 {
                            moe: ref_moe_view(l, Slot::Q8Moe)?,
                            down: ref_moe_view(l, Slot::Q8Down)?,
                            shared: ref_moe_view(l, Slot::Q8Shared)?,
                        },
                        expert_gu: ref_view(Slot::ExpertGu)?,
                        expert_down: ref_view(Slot::ExpertDown)?,
                        shared_gu: ref_view(Slot::SharedGu)?,
                        shared_out: ref_view(Slot::SharedOut)?,
                    },
                };
                run_moe_block(
                    &Launch {
                        ctx: ctx0,
                        stream: stream0,
                    },
                    moe,
                    &ref_moe_set,
                    &moe_full_dims,
                )?;
            }
            let norm = norm_ref.range(0, norm_ref.bytes())?;
            ctx0.add_rmsnorm_q8(
                stream0,
                AddNormQ8Call {
                    n: hidden,
                    eps,
                    h: ref_view(final_resid(layers))?,
                    delta: None,
                    weight: norm,
                    norm_out: ref_view(Slot::FinalNormRow)?,
                    q8: ref_view(Slot::Q8Final)?,
                },
            )?;
            ctx0.matvec_q(
                stream0,
                MatvecCall {
                    ty: head_ref.quant(),
                    w: head_ref.buffer().range(0, head_ref.buffer().bytes())?,
                    x_q8: ref_view(Slot::Q8Final)?,
                    y: ref_view(Slot::Logits)?,
                    k: head_ref.k(),
                    rows: head_ref.n_rows(),
                    ncols: 1,
                },
            )?;
            ctx0.argmax(
                stream0,
                ArgmaxCall {
                    n: head_ref.n_rows(),
                    logits: ref_view(Slot::Logits)?,
                    part: ref_view(Slot::ArgmaxPart)?,
                    params,
                    result: Some(state_ref.word(Word::ResultPair)?),
                },
            )?;
            ctx0.step_advance(stream0, params, 1)?;
            stream0.synchronize()?;
            let pair = {
                let word = state_ref.word(Word::ResultPair)?;
                let mut bytes = [0u8; 8];
                word.buffer().read_at(stream0, word.offset(), &mut bytes)?;
                (
                    i32::from_le_bytes(bytes[..4].try_into().unwrap()),
                    i32::from_le_bytes(bytes[4..].try_into().unwrap()),
                )
            };
            let logits_view = ref_view(Slot::Logits)?;
            let logits = {
                let mut bytes = vec![0u8; logits_view.bytes()];
                logits_view
                    .buffer()
                    .read_at(stream0, logits_view.offset(), &mut bytes)?;
                bytes_f32(&bytes)
            };
            let boundaries = (0..layers)
                .map(|l| {
                    let view = ref_view(resid_out(l))?;
                    let mut bytes = vec![0u8; view.bytes()];
                    view.buffer().read_at(stream0, view.offset(), &mut bytes)?;
                    Ok(bytes_f32(&bytes))
                })
                .collect::<Result<_, EngError>>()?;
            // The production cycle ends with the commit of the read pair.
            state_ref
                .commit_step(1, pair.1)
                .map_err(|e| EngError::Checkpoint(format!("ref commit: {e:?}")))?;
            (pair.0, pair.1, logits, boundaries)
        };

        // The distributed step.
        run_card_token(
            &stack,
            &states,
            &kinds,
            &ordinals,
            &block_ops_refs,
            &kv_runs,
            &mixers,
            &moe_refs,
            &embed_ref,
            &output_norms,
            &heads,
            eps,
            t,
            true,
        )?;
        let mut card_reads = Vec::with_capacity(2);
        let mut boundaries = Vec::with_capacity(2);
        for (card, state) in states.iter().enumerate() {
            stack.streams()[card].synchronize()?;
            let pair = {
                let word = state.word(Word::ResultPair)?;
                let mut bytes = [0u8; 8];
                word.buffer()
                    .read_at(&stack.streams()[card], word.offset(), &mut bytes)?;
                (
                    i32::from_le_bytes(bytes[..4].try_into().unwrap()),
                    i32::from_le_bytes(bytes[4..].try_into().unwrap()),
                )
            };
            let logits_view = card_view(card, Slot::Logits)?;
            let shard = {
                let mut bytes = vec![0u8; logits_view.bytes()];
                logits_view.buffer().read_at(
                    &stack.streams()[card],
                    logits_view.offset(),
                    &mut bytes,
                )?;
                bytes_f32(&bytes)
            };
            card_reads.push((pair.0, pair.1, shard));
            let mut per_layer = Vec::with_capacity(layers);
            for l in 0..layers {
                let view = card_view(card, resid_out(l))?;
                let mut bytes = vec![0u8; view.bytes()];
                view.buffer()
                    .read_at(&stack.streams()[card], view.offset(), &mut bytes)?;
                per_layer.push(bytes_f32(&bytes));
            }
            boundaries.push(per_layer);
        }
        let starts: Vec<usize> = head_runs.iter().map(|r| r.start).collect();
        let shards: Vec<&[f32]> = card_reads.iter().map(|(_, _, s)| s.as_slice()).collect();
        let merged_id = merge_head_candidates(&shards, &starts);
        let (ref_best, ref_runner) = top2_of([ref_logits.as_slice()]);
        let (merged_best, merged_runner) = top2_of(shards.iter().copied());
        steps.push(StepEvidence {
            ref_id,
            ref_status,
            merged_id,
            cards: card_reads,
            boundaries,
            ref_boundaries,
            ref_logits,
            ref_margin: ref_best - ref_runner,
            merged_margin: merged_best - merged_runner,
        });
        // The greedy chain. The cards' own argmax wrote each card's
        // LOCAL id into its TokenId word, so the merged global id is
        // injected explicitly; `pos` and `kv_len` keep moving through
        // each card's `step_advance`. The reference reads its own
        // StepParams word, so teacher-forced mode injects the merged id
        // there too by feeding the next write, and independent mode
        // leaves the reference on its own argmax.
        ref_token = if independent {
            ref_id as u32
        } else {
            merged_id.max(0) as u32
        };
        card_token = merged_id.max(0) as u32;
        if t + 1 < tokens {
            for (card, state) in states.iter().enumerate() {
                let stream = &stack.streams()[card];
                let word = state.step_field(StepField::TokenId)?;
                word.buffer().write_at(
                    stream,
                    word.offset(),
                    &(card_token as i32).to_le_bytes(),
                )?;
            }
        }
    }

    // The ring words, every Full layer's KV planes, and every DeltaNet
    // layer's committed state.
    let stream0 = &stack.streams()[0];
    let mut ring_words = Vec::with_capacity(2);
    for card in 0..2 {
        let mut word = [0u8; 4];
        stack.shards()[card]
            .ring_word()
            .read_at(&stack.streams()[card], 0, &mut word)?;
        ring_words.push(u32::from_le_bytes(word));
    }
    let mut kv_drift = Vec::with_capacity(2);
    for (card, state) in states.iter().enumerate() {
        let mut per_kv = Vec::with_capacity(full_ordinals.len());
        for &ord in &full_ordinals {
            let mut planes = Vec::with_capacity(4);
            for plane in [Plane::K, Plane::KScales, Plane::V, Plane::VScales] {
                let want = {
                    let view = state_ref.kv_plane(ord, plane)?;
                    let mut bytes = vec![0u8; view.bytes()];
                    view.buffer().read_at(stream0, view.offset(), &mut bytes)?;
                    bytes
                };
                let got = {
                    let view = state.kv_plane(ord, plane)?;
                    let mut bytes = vec![0u8; view.bytes()];
                    view.buffer()
                        .read_at(&stack.streams()[card], view.offset(), &mut bytes)?;
                    bytes
                };
                planes.push(PlaneDrift::of(&want, &got));
            }
            per_kv.push(planes);
        }
        kv_drift.push(per_kv);

        // Slot 0 is the committed DeltaNet state. The card's state is
        // COMPACTED: its dealt value heads sit at local indices and its
        // conv channels are its segments in stored order - the mapping
        // the block rig compares bit-exactly at identical inputs. Over
        // the whole token chain the q8 quantize absorbs most of the
        // residual envelope drift, but a value crossing a bin boundary
        // shifts its whole head block, so the gate here is the state
        // envelope, not bit equality.
        let head_block = d.shape.delta.key_dim * d.shape.delta.value_dim;
        let conv_taps = d.shape.delta.conv_kernel - 1;
        let head_run = table.value_heads(card)?;
        for &ord in &delta_ordinals {
            let rec_ref = {
                let rec = state_ref.delta_rec(0, ord)?;
                let mut rb = vec![0u8; rec.bytes()];
                rec.buffer().read_at(stream0, rec.offset(), &mut rb)?;
                bytes_f32(&rb)
            };
            let conv_ref = {
                let conv = state_ref.delta_conv(0, ord)?;
                let mut cb = vec![0u8; conv.bytes()];
                conv.buffer().read_at(stream0, conv.offset(), &mut cb)?;
                bytes_f32(&cb)
            };
            let (rec_card, conv_card) = {
                let state = &states[card];
                let rec = state.delta_rec(0, ord)?;
                let conv = state.delta_conv(0, ord)?;
                let mut rb = vec![0u8; rec.bytes()];
                rec.buffer()
                    .read_at(&stack.streams()[card], rec.offset(), &mut rb)?;
                let mut cb = vec![0u8; conv.bytes()];
                conv.buffer()
                    .read_at(&stack.streams()[card], conv.offset(), &mut cb)?;
                (bytes_f32(&rb), bytes_f32(&cb))
            };
            let want = &rec_ref[head_run.start * head_block..head_run.end * head_block];
            let got = &rec_card[..want.len()];
            within_state_envelope(&format!("delta rec (ordinal {ord})"), card, got, want);
            let full_channels = conv_ref.len() / conv_taps;
            let mut conv_want = Vec::with_capacity(conv_card.len());
            for tap in 0..conv_taps {
                for seg in table.delta_qkv_segments(card)? {
                    conv_want.extend_from_slice(
                        &conv_ref[tap * full_channels + seg.start..tap * full_channels + seg.end],
                    );
                }
            }
            within_state_envelope(
                &format!("delta conv (ordinal {ord})"),
                card,
                &conv_card[..conv_want.len()],
                &conv_want,
            );
        }
    }
    Ok(Evidence {
        steps,
        ring_words,
        full_ordinals,
        kv_drift,
        starts: head_runs.iter().map(|r| r.start).collect(),
    })
}

#[test]
fn a_whole_token_runs_distributed_and_matches_the_reference() {
    const NAME: &str = "a_whole_token_runs_distributed_and_matches_the_reference";
    if !device_present(NAME) {
        return;
    }
    let evidence = run_rig(rig_spec(), TOKENS, false).expect("rig");
    assert_reference_match(&evidence, TOKENS);
}

/// Every gate the whole-model tests share: sequence equality (the only
/// id the cards ever see is their own merged one), status words,
/// per-layer residual envelopes, logit-shard envelopes, the greedy
/// winner's separation, ring health, and end-of-run KV and DeltaNet
/// state planes.
fn assert_reference_match(evidence: &Evidence, tokens: usize) {
    assert_eq!(evidence.steps.len(), tokens, "one step per token");
    for (t, step) in evidence.steps.iter().enumerate() {
        assert_eq!(
            step.merged_id, step.ref_id,
            "token {t}: the merged greedy id must equal the reference argmax"
        );
        assert_eq!(step.ref_status, 0, "token {t}: reference status");
        assert!(
            step.ref_id >= 0,
            "token {t}: the chain needs a comparable id"
        );
        for (card, (local, status, shard)) in step.cards.iter().enumerate() {
            assert_eq!(*status, 0, "token {t} card {card}: argmax status");
            assert_eq!(
                *local,
                host_argmax(shard),
                "token {t} card {card}: the device argmax over its own shard"
            );
            let start = evidence.starts[card];
            let want = &step.ref_logits[start..start + shard.len()];
            within_envelope(&format!("logits shard (token {t})"), card, shard, want);
            for (l, got) in step.boundaries[card].iter().enumerate() {
                within_envelope(
                    &format!("resid_out({l}) (token {t})"),
                    card,
                    got,
                    &step.ref_boundaries[l],
                );
            }
        }
        // The merged margin must track the reference margin: the winner
        // and the runner-up each carry the logit envelope.
        let peak = step.ref_logits.iter().fold(0f32, |a, &v| a.max(v.abs()));
        let slack = 2.0 * (peak * 1e-3 + 1e-6);
        assert!(
            (step.merged_margin - step.ref_margin).abs() <= slack,
            "token {t}: merged margin {} vs reference {}",
            step.merged_margin,
            step.ref_margin
        );
        assert!(
            step.ref_margin > 0.0,
            "token {t}: the winner must be separated (margin {})",
            step.ref_margin
        );
    }
    for (card, &word) in evidence.ring_words.iter().enumerate() {
        assert_eq!(word, 0, "card {card}: a ring exchange failed");
    }
    for (card, per_ordinal) in evidence.kv_drift.iter().enumerate() {
        for (i, planes) in per_ordinal.iter().enumerate() {
            for (p, drift) in planes.iter().enumerate() {
                assert!(
                    drift.within_bound(),
                    "card {card} full ordinal {} plane {p}: {} of {} bytes differ (max delta {})",
                    evidence.full_ordinals[i],
                    drift.diffs,
                    drift.bytes,
                    drift.max_delta
                );
            }
        }
    }
}

#[test]
fn a_whole_40_layer_model_runs_distributed_and_matches_the_reference() {
    const NAME: &str = "a_whole_40_layer_model_runs_distributed_and_matches_the_reference";
    if !device_present(NAME) {
        return;
    }
    // Independent greedy continuation: the reference feeds back its own
    // argmax and the cards their merged id, so sequence equality proves
    // the coordinator's real feedback mechanism, not just numerical
    // agreement on identical inputs.
    let evidence = run_rig(real_layer_spec(), TOKENS, true).expect("rig");
    assert_reference_match(&evidence, TOKENS);
}
