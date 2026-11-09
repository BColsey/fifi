//! The multi-device chunk driver: one prefill chunk through BOTH cards.
//!
//! The stage functions are the single-device chunk pipeline's, driven per
//! card over card-built [`ChunkRun`]s ([`ChunkModel::card`], the card
//! geometry, the card's inter slices). Three ring operations per layer
//! make the cards' partial rows whole at the exact points a card's view
//! is narrower than the model's:
//!
//! 1. after the mixer stages, `Delta` holds the card's PARTIAL mixer
//!    output; one `exchange_sum` of `m * hidden` values gives both cards
//!    the full row `NormF` consumes.
//! 2. routing runs on card 0 and TWO/THREE small broadcasts carry the
//!    picks (`TopkId`, `TopkWeight`, `SharedGate`) so the cards can never
//!    disagree on a pick.
//! 3. after the MoE block, card 0's combine accumulated its partial into
//!    `Hidden` (residual included) while card 1 combined `To(Delta)`
//!    reading the permanent `Zero` row as its residual - one
//!    `exchange_sum` over the two slots writes
//!    `residual + sum(partials)` into both cards' `Hidden`.
//!
//! The embedding runs on card 0 (the table lives there) and one broadcast
//! carries the embedded rows to card 1's `Hidden`.
//!
//! Chunk rows are capped at `m * hidden <= MAX_VALUES` (512 rows at
//! hidden 2048): every exchange of a chunk is one ring sequence. The
//! driver is enqueue-only until the caller folds the status words.

use super::{
    card_step::CardStack,
    dims::StepDims,
    layer_table::LayerTable,
    mixer_geometry::MixerGeometry,
    moe::{Launch, MoeLayerWeights},
    prefill::ChunkRows,
    prefill::{
        ChunkBufs, ChunkModel, ChunkRun, ChunkViews, ChunkWeights, LayerCtx, LayerMoe, Stage,
        attn_flash, attn_out, attn_prep, attn_proj, chunk_embed, chunk_route, combine_chunk_moe,
        delta_conv, delta_proj, delta_scan, layer_stages, norm_a, norm_f, quant_a, quant_a_out,
        quant_d_out, run_chunk_routed, run_chunk_shared, ssm_out,
    },
    prefill_driver::PrefillArena,
    prefill_plan::{ChunkSlot, chunk_tensors_for},
    state::ResidentState,
    state_layout::Word,
};
use crate::{
    EngError,
    family::qwen36::host::MoeDims as HostMoeDims,
    kernels::cuda::NativeWeight,
    kernels::cuda::peer_ring::{MAX_VALUES, RingMode},
};

/// The largest chunk rows the driver accepts for this shape: every layer
/// sum is one ring sequence, and the transport caps a payload at
/// [`MAX_VALUES`] values.
pub(crate) fn max_chunk_rows(hidden: usize) -> usize {
    (MAX_VALUES / hidden).max(1)
}

/// One card's chunk resources: its arena (the card's narrower slot plan)
/// and its chunk model (the card's weight views under the card geometry).
struct CardChunkCard {
    arena: PrefillArena,
    model: ChunkModel<'static, 'static>,
    moe: &'static [MoeLayerWeights<'static>],
}

/// The session's chunk resources, built once at start.
pub(crate) struct CardChunk {
    cards: Vec<CardChunkCard>,
    m_sess: usize,
    hidden: usize,
}

/// Builds both cards' chunk arenas and models. `m_chunk` is the chunk row
/// count (the caller checked `m_chunk <= max_chunk_rows(hidden)`; v1 uses
/// one fixed size, not the ladder).
/// The session's 'static weight views the chunk models borrow.
pub(crate) struct CardChunkInputs {
    /// The per-card loaded weights, leaked at start (`&'static [&'static CudaWeights]`).
    pub(crate) weights: &'static [&'static crate::family::project::cuda::CudaWeights],
    /// The embedding tensor's name in card 0's weights.
    pub(crate) embed_name: &'static str,
}

pub(super) fn build_card_chunk(
    stack: &'static CardStack,
    dims: &StepDims,
    table: &LayerTable,
    inputs: CardChunkInputs,
    options: &super::dims::StepOptions,
    m_chunk: usize,
) -> Result<CardChunk, EngError> {
    let cards = stack.shards().len();
    let hidden = dims.shape.hidden;
    if m_chunk == 0 || m_chunk > max_chunk_rows(hidden) {
        return Err(EngError::BadRequest(format!(
            "card chunk: {m_chunk} rows is outside 1..={}",
            max_chunk_rows(hidden)
        )));
    }
    let mut out = Vec::with_capacity(cards);
    for card in 0..cards {
        let context = &stack.contexts()[card];
        let geometry = MixerGeometry::for_card(dims, stack.table(), card)?;
        let host_moe = HostMoeDims {
            inter: stack.table().expert_inter(card)?.len(),
            shared_inter: stack.table().shared_inter(card)?.len(),
            ..dims.shape.moe
        };
        let facts = context.mmq_device_facts()?;
        let list = chunk_tensors_for(dims, &geometry, &host_moe, m_chunk, facts.nsm)?;
        // The card's arena: the same builder the single-device path uses,
        // over the card's list (zero-filled, part buffer, kv8 staging).
        let arena = PrefillArena::allocate_list(
            &stack.contexts()[card],
            &stack.streams()[card],
            list,
            (m_chunk, facts, m_chunk * hidden * 4),
        )?;
        // The card's own views, resolved exactly as the session resolves
        // them, then leaked for the chunk models' process lifetime.
        let weights_card_of = inputs.weights[card];
        let wanted: Vec<usize> = (0..table.len()).collect();
        let weights_card = super::mixer_weights::MixerWeights::new(
            &geometry,
            table,
            weights_card_of,
            &wanted,
            options.layout,
        )?;
        let moe_specs = super::resources::MoeSpecs::for_card(dims, stack.table(), card)?;
        let moe_card: &'static [MoeLayerWeights<'static>] = Box::leak(
            (0..table.len())
                .map(|layer| {
                    super::resources::moe_layer_weights(
                        table,
                        weights_card_of,
                        &moe_specs,
                        layer,
                        options.layout,
                    )
                })
                .collect::<Result<Vec<_>, EngError>>()?
                .into_boxed_slice(),
        );
        // Card 0 holds the embedding table; the other cards' models
        // reference it through card 0's leaked weights.
        let Some(crate::family::project::cuda::CudaWeight::Native { weight, .. }) =
            inputs.weights[0].get(inputs.embed_name)
        else {
            return Err(EngError::Checkpoint(format!(
                "card chunk: {} is not a native tensor",
                inputs.embed_name
            )));
        };
        let embed: &'static NativeWeight = weight;
        let weights = ChunkWeights::new(Box::leak(Box::new(weights_card)), moe_card, embed);
        let model = ChunkModel::card(
            Box::leak(Box::new(table.clone())),
            Box::leak(Box::new(geometry)),
            dims,
            weights,
            host_moe.inter,
            host_moe.shared_inter,
        )?;
        out.push(CardChunkCard {
            arena,
            model,
            moe: moe_card,
        });
    }
    Ok(CardChunk {
        cards: out,
        m_sess: m_chunk,
        hidden,
    })
}

/// Runs one chunk of `tokens.len()` rows at base position `p0` through
/// both cards. Enqueue-only: the caller syncs the streams and folds the
/// status words after this returns; a refusal before the first launch
/// leaves the states untouched.
pub(crate) fn run_card_chunk(
    stack: &CardStack,
    states: &mut [ResidentState],
    chunk: &CardChunk,
    tokens: &[u32],
    p0: usize,
) -> Result<(), EngError> {
    let cards = stack.shards().len();
    let m = tokens.len();
    let hidden = chunk.hidden;
    if m == 0 || m > chunk.m_sess || m > max_chunk_rows(hidden) {
        return Err(EngError::BadRequest(format!(
            "card chunk: {m} rows is outside 1..={}",
            chunk.m_sess.min(max_chunk_rows(hidden))
        )));
    }
    let capacity = states[0].layout().kv.capacity;
    if p0 + m > capacity {
        return Err(EngError::ContextLimit(format!(
            "card chunk: p0 {p0} + m {m} passes the context cap {capacity}"
        )));
    }
    let mut launches = Vec::with_capacity(cards);
    for card in 0..cards {
        launches.push(Launch {
            ctx: &stack.contexts()[card],
            stream: &stack.streams()[card],
        });
    }
    // The state prelude before any immutable borrow: begin and clear on
    // every card, then the runs borrow the states immutably.
    for (card, state) in states.iter_mut().enumerate() {
        state.begin_prefill()?;
        state.clear_prefill_status(&stack.streams()[card])?;
    }
    let mut views = Vec::with_capacity(cards);
    let mut bufs = Vec::with_capacity(cards);
    for card in 0..cards {
        let v = ChunkViews::new(
            chunk.cards[card].arena.list(),
            chunk.cards[card].arena.arena(),
        )?;
        let b = ChunkBufs::new(
            v.clone(),
            chunk.cards[card]
                .arena
                .part()
                .range(0, chunk.cards[card].arena.part_bytes())?,
            chunk.cards[card].arena.facts(),
            chunk.m_sess,
        )?
        .with_kv8_staging(chunk.cards[card].arena.kv8_staging()?);
        // The ids upload is the one blocking write per chunk, as in the
        // single-device driver.
        let ids = v.get(ChunkSlot::IdsTok)?;
        let bytes: Vec<u8> = tokens.iter().flat_map(|id| id.to_le_bytes()).collect();
        ids.buffer()
            .write_at(&stack.streams()[card], ids.offset(), &bytes)?;
        views.push(v);
        bufs.push(b);
    }
    // The runs borrow the states, the models and this call's bufs under
    // one lifetime; everything they touch outlives the runs vector.
    let mut runs = Vec::with_capacity(cards);
    for card in 0..cards {
        let rows = ChunkRows::new(p0, m, capacity, chunk.m_sess)?;
        runs.push(ChunkRun::new(
            &states[card],
            &chunk.cards[card].model,
            &bufs[card],
            rows,
        )?);
    }
    // The embedded rows: card 0 embeds into Hidden, one broadcast to the
    // other cards' Hidden.
    let status: Vec<_> = (0..cards)
        .map(|card| states[card].word(Word::PrefillStatus))
        .collect::<Result<_, EngError>>()?;
    chunk_embed(&launches[0], &runs[0], status[0])?;
    if cards > 1 {
        let src = views[0].get(ChunkSlot::Hidden)?;
        let outs: Vec<_> = (0..cards)
            .map(|card| views[card].get(ChunkSlot::Hidden))
            .collect::<Result<_, EngError>>()?;
        stack.broadcast(0, src, &outs, &status, m * hidden)?;
    }
    // The layer walk.
    let table = chunk.cards[0].model.table();
    for layer in 0..table.len() {
        let slot = table.slot(layer)?;
        run_card_chunk_layer(
            &LayerDrive {
                stack,
                chunk,
                runs: &runs,
                views: &views,
                launches: &launches,
                status: &status,
                m,
                hidden,
            },
            layer,
            slot.kind,
            slot.ordinal,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
struct LayerDrive<'a, 'b, 'r, 'w>
where
    'b: 'r,
{
    stack: &'a CardStack,
    chunk: &'a CardChunk,
    runs: &'a [ChunkRun<'r, 'w>],
    views: &'a [ChunkViews<'b>],
    launches: &'a [Launch<'a>],
    status: &'a [crate::kernels::cuda::DeviceRange<'b>],
    m: usize,
    hidden: usize,
}

fn run_card_chunk_layer(
    drive: &LayerDrive<'_, '_, '_, '_>,
    layer: usize,
    kind: crate::family::AttentionKind,
    ordinal: usize,
) -> Result<(), EngError> {
    let LayerDrive {
        stack,
        chunk,
        runs,
        views,
        launches,
        status,
        m,
        hidden,
    } = drive;
    let (m, hidden) = (*m, *hidden);
    let cards = stack.shards().len();
    let stages = layer_stages(kind)?;
    for stage in stages {
        match stage {
            Stage::NormF => {
                // The mixer output exchange precedes the norm that reads it.
                exchange_slot(stack, views, status, ChunkSlot::Delta, m * hidden)?;
                for card in 0..cards {
                    let ctx = layer_ctx(launches, chunk, status, card, layer, kind, ordinal)?;
                    norm_f(&runs[card], &ctx)?;
                }
            }
            Stage::Route => {
                // Card 0 routes; the picks broadcast to every card.
                {
                    let ctx = layer_ctx(launches, chunk, status, 0, layer, kind, ordinal)?;
                    chunk_route(&runs[0], &ctx)?;
                }
                for slot in [
                    ChunkSlot::TopkId,
                    ChunkSlot::TopkWeight,
                    ChunkSlot::SharedGate,
                ] {
                    let src = views[0].get(slot)?;
                    let outs: Vec<_> = (0..cards)
                        .map(|card| views[card].get(slot))
                        .collect::<Result<_, EngError>>()?;
                    let count = match slot {
                        ChunkSlot::TopkId | ChunkSlot::TopkWeight => {
                            m * chunk.cards[0].model.moe_dims().top_k
                        }
                        _ => m,
                    };
                    stack.broadcast(0, src, &outs, status, count)?;
                }
            }
            Stage::Moe => {
                // The shared expert and the routed GEMMs per card, then
                // the asymmetric combine and the Hidden exchange.
                for card in 0..cards {
                    let io = moe_io(&runs[card], &status[card], m)?;
                    run_chunk_shared(&launches[card], &chunk.cards[card].moe[layer], &io)?;
                    run_chunk_routed(&launches[card], &chunk.cards[card].moe[layer], &io)?;
                    if card == 0 {
                        combine_chunk_moe(&launches[card], &io)?;
                    } else {
                        // The partial alone: combine into the dead Delta
                        // row reading the permanent Zero row as residual.
                        let zero = views[card].get(ChunkSlot::Zero)?;
                        let delta = views[card].get(ChunkSlot::Delta)?;
                        launches[card].ctx.moe_combine(
                            launches[card].stream,
                            crate::kernels::cuda::resident::moe_combine::CombineCall {
                                shape: super::prefill::chunk_combine_shape(&io.dims, m),
                                res: zero,
                                down: views[card].get(ChunkSlot::Down)?,
                                weights: views[card].get(ChunkSlot::TopkWeight)?,
                                shared: Some(
                                    crate::kernels::cuda::resident::moe_combine::SharedTerm {
                                        gate: views[card].get(ChunkSlot::SharedGate)?,
                                        row: views[card].get(ChunkSlot::ShOut)?,
                                    },
                                ),
                                out: crate::kernels::cuda::resident::moe_combine::CombineOut::To(
                                    delta,
                                ),
                            },
                        )?;
                    }
                }
                // Card 0 publishes Hidden (residual + partial), the others
                // Delta (partial alone): the sum is the correct new row.
                let partials: Vec<_> = (0..cards)
                    .map(|card| {
                        views[card].get(if card == 0 {
                            ChunkSlot::Hidden
                        } else {
                            ChunkSlot::Delta
                        })
                    })
                    .collect::<Result<_, EngError>>()?;
                let outs: Vec<_> = (0..cards)
                    .map(|card| views[card].get(ChunkSlot::Hidden))
                    .collect::<Result<_, EngError>>()?;
                stack.exchange_sum(&partials, &outs, status, m * hidden, RingMode::F32)?;
            }
            other => {
                // The card-local stages, driven on every card.
                for card in 0..cards {
                    let ctx = layer_ctx(launches, chunk, status, card, layer, kind, ordinal)?;
                    match other {
                        Stage::NormA => norm_a(&runs[card], &ctx)?,
                        Stage::QuantA => quant_a(&runs[card], &ctx)?,
                        Stage::DeltaProj => delta_proj(&runs[card], &ctx)?,
                        Stage::Conv => delta_conv(&runs[card], &ctx)?,
                        Stage::Scan => delta_scan(&runs[card], &ctx)?,
                        Stage::QuantDOut => quant_d_out(&runs[card], &ctx)?,
                        Stage::SsmOut => ssm_out(&runs[card], &ctx)?,
                        Stage::AttnProj => attn_proj(&runs[card], &ctx)?,
                        Stage::Prep => attn_prep(&runs[card], &ctx)?,
                        Stage::Flash => attn_flash(&runs[card], &ctx)?,
                        Stage::QuantAOut => quant_a_out(&runs[card], &ctx)?,
                        Stage::AttnOut => attn_out(&runs[card], &ctx)?,
                        Stage::NormF | Stage::Route | Stage::Moe => unreachable!("handled above"),
                    }
                }
            }
        }
    }
    Ok(())
}

fn layer_ctx<'a, 'b, 'w>(
    launches: &'a [Launch<'a>],
    chunk: &'a CardChunk,
    status: &'b [crate::kernels::cuda::DeviceRange<'b>],
    card: usize,
    layer: usize,
    kind: crate::family::AttentionKind,
    ordinal: usize,
) -> Result<LayerCtx<'a, 'w>, EngError>
where
    'b: 'a,
{
    let weights = chunk.cards[card]
        .model
        .weights()
        .mixer()
        .layer_weights(layer)?;
    let moe = LayerMoe::Native(&chunk.cards[card].moe[layer]);
    Ok(LayerCtx {
        launch: &launches[card],
        weights,
        moe,
        ord: ordinal,
        layer,
        kind,
        status: status[card],
    })
}

fn exchange_slot(
    stack: &CardStack,
    views: &[ChunkViews<'_>],
    status: &[crate::kernels::cuda::DeviceRange<'_>],
    slot: ChunkSlot,
    count: usize,
) -> Result<(), EngError> {
    let partials: Vec<_> = (0..views.len())
        .map(|card| views[card].get(slot))
        .collect::<Result<_, EngError>>()?;
    let outs = partials.clone();
    stack.exchange_sum(&partials, &outs, status, count, RingMode::F32)
}

fn moe_io<'b>(
    run: &'b ChunkRun<'_, '_>,
    status: &'b crate::kernels::cuda::DeviceRange<'b>,
    m: usize,
) -> Result<super::prefill::MoeChunkIo<'b>, EngError> {
    Ok(super::prefill::MoeChunkIo {
        m,
        dims: run.model().model_dims_at(m),
        x2: run.v(ChunkSlot::X2)?,
        ids: run.v(ChunkSlot::TopkId)?,
        weights: run.v(ChunkSlot::TopkWeight)?,
        shared_gate: run.v(ChunkSlot::SharedGate)?,
        h: run.v(ChunkSlot::Hidden)?,
        status: *status,
        scratch: run.views(),
    })
}
