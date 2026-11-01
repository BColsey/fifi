//! The eager two-card token driver: one whole decode token across every
//! card - the assembly of the passing block drivers into a model step.
//!
//! The op order is the single-device `step_ops` list with the ring markers
//! of `multi.rs` in place: `step_begin`; card 0 embeds and the embedded
//! row rides ONE broadcast (`EmbedBroadcast`) so every card's residual
//! stream starts identical; per layer the mixer block (`run_card_block`,
//! its K/V gathers inside) then the MoE block (`run_card_moe_block`, its
//! route broadcast and sums inside); the final norm, the head projection
//! over the card's own vocabulary shard, the argmax over that shard, and
//! `step_advance` - every card, on its own stream.
//!
//! The residual stream alternates exactly as single-device `resid_in`/
//! `resid_out` pin: the embedding writes `ResidA`, layer `l`'s mixer adds
//! into `resid_in(l)` in place, its MoE writes `resid_out(l)`, the final
//! norm reads `final_resid(layers)`.
//!
//! The greedy token is MERGED ON THE HOST: each card's argmax covers its
//! own `head_rows` shard (the card plan's `Logits` slot holds only those
//! rows), and [`merge_head_candidates`] reproduces the native argmax
//! contract over the concatenated shards - largest value, lowest GLOBAL
//! id on a tie, a NaN never wins, all-NaN gives -1.
//!
//! The v1 failure model of `multi.rs` holds: the K/V gather pair is gated
//! host-side before the append (inside `run_card_block`); every other
//! exchange folds at the token's end - the caller reads every card's ring
//! word and pair before accepting the token.

use crate::EngError;
use crate::family::qwen36::cuda::card_block::{run_card_block, run_card_moe_block};
use crate::family::qwen36::cuda::card_step::CardStack;
use crate::family::qwen36::cuda::mixer::Mixers;
use crate::family::qwen36::cuda::mixer_ops::CardOp;
use crate::family::qwen36::cuda::moe::{
    CardMoeOp, ExpertBufs, MoeBuffers, MoeLayerWeights, Packs, RouteBufs, moe_sequence_card,
};
use crate::family::qwen36::cuda::state::StepField;
use crate::family::qwen36::cuda::state_layout::Word;
use crate::family::qwen36::cuda::step_ops::{final_resid, resid_in, resid_out};
use crate::family::qwen36::cuda::step_plan::Slot;
use crate::kernels::cuda::DeviceBuffer;
use crate::kernels::cuda::DeviceRange;
use crate::kernels::cuda::NativeWeight;
use crate::kernels::cuda::resident::add_rmsnorm_q8::AddNormQ8Call;
use crate::kernels::cuda::resident::embed::EmbedNativeCall;
use crate::kernels::cuda::resident::matvec_q::MatvecCall;
use crate::kernels::cuda::resident::step_params::ArgmaxCall;

const F32: usize = size_of::<f32>();

/// The global greedy token of the cards' head-shard candidates: `shards`
/// are the shard logits rows in ascending `starts` order. The rule is the
/// native argmax contract over the concatenated row - largest value, the
/// lowest index on a tie, a NaN never wins, -1 when no value is
/// comparable - so a two-card merge equals the one-card argmax whenever
/// the shard rows equal the reference row's ranges.
pub(crate) fn merge_head_candidates(shards: &[&[f32]], starts: &[usize]) -> i32 {
    let mut best: Option<(usize, f32)> = None;
    for (shard, &start) in shards.iter().zip(starts) {
        for (i, &value) in shard.iter().enumerate() {
            if value.is_nan() {
                continue;
            }
            let better = match best {
                // Scanning ascending global ids, only a STRICTLY larger
                // value replaces the winner: ties keep the lower id.
                Some((_, top)) => value > top,
                None => true,
            };
            if better {
                best = Some((start + i, value));
            }
        }
    }
    best.map(|(id, _)| id as i32).unwrap_or(-1)
}

/// Runs one whole decode token on every card. `kinds`/`ordinals` are the
/// layer table's kinds and state ordinals in layer order; `mixers[card]`
/// holds that card's mixer parts for every layer; `moe[layer][card]` its
/// MoE weights; `block_ops[layer][card]` its mixer card op list
/// (`CardKindOps::of`); `kv_runs[card]` its absolute K/V row range;
/// `embed` card 0's native table; `output_norms[card]` and `heads[card]`
/// the replicated final-norm row and the card's vocabulary-shard head;
/// `admit_pos` the caller's committed row count (the token book is the
/// CALLER's: write the step params of token 0 with
/// `CardStack::write_step_params` before the first call, and for every
/// later token write the merged token id into card 0's `TokenId` word -
/// each card's own argmax writes its LOCAL id, which is not the global
/// one - and let `step_advance`, run at each token's end, keep `pos` and
/// `kv_len` moving).
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_card_token(
    stack: &CardStack,
    kinds: &[crate::family::AttentionKind],
    ordinals: &[usize],
    block_ops: &[&[&[CardOp]]],
    kv_runs: &[std::ops::Range<usize>],
    mixers: &[Mixers<'_>],
    moe: &[&[MoeLayerWeights<'_>]],
    embed: &NativeWeight,
    output_norms: &[DeviceBuffer],
    heads: &[NativeWeight],
    eps: f32,
    admit_pos: usize,
) -> Result<(), EngError> {
    let cards = stack.shards().len();
    if kinds.len() != ordinals.len()
        || kinds.len() != block_ops.len()
        || kinds.len() != moe.len()
        || mixers.len() != cards
        || kv_runs.len() != cards
        || output_norms.len() != cards
        || heads.len() != cards
    {
        return Err(EngError::BadRequest(format!(
            "card token: {} layers, {} ordinals, {} op lists, {} moe sets, \
             {cards} cards, {} kv runs, {} norms, {} heads",
            kinds.len(),
            ordinals.len(),
            block_ops.len(),
            moe.len(),
            kv_runs.len(),
            output_norms.len(),
            heads.len()
        )));
    }
    for set in moe.iter() {
        if set.len() != cards {
            return Err(EngError::BadRequest(format!(
                "card token: a moe layer has {} weight sets for {cards} cards",
                set.len()
            )));
        }
    }
    for list in block_ops.iter() {
        if list.len() != cards {
            return Err(EngError::BadRequest(format!(
                "card token: a layer has {} op lists for {cards} cards",
                list.len()
            )));
        }
    }
    let view = |card: usize, slot: Slot| -> Result<DeviceRange<'_>, EngError> {
        let at = stack.plans()[card]
            .index(slot)
            .ok_or_else(|| EngError::Checkpoint(format!("card token: no {slot:?} slot")))?;
        stack.shards()[card].tensor(at)
    };
    let hidden = view(0, Slot::ResidA)?.bytes() / F32;
    let statuses: Vec<DeviceRange<'_>> = (0..cards)
        .map(|card| stack.shards()[card].state().step_status())
        .collect::<Result<_, _>>()?;
    let ring_statuses: Vec<DeviceRange<'_>> = (0..cards)
        .map(|card| stack.shards()[card].ring_status())
        .collect::<Result<_, _>>()?;

    // The MoE buffers of parity `p` on `card`, viewing the card plan's own
    // slots - the same wiring `step.rs::moe_buffers` gives the one-device
    // step, at the card's narrowed intermediate widths.
    let moe_bufs = |parity: usize| -> Result<Vec<MoeBuffers<'_>>, EngError> {
        let mut sets = Vec::with_capacity(cards);
        for card in 0..cards {
            let route = RouteBufs {
                resid_in: view(card, resid_in(parity))?,
                resid_out: view(card, resid_out(parity))?,
                norm_row: view(card, Slot::NormRow)?,
                logits: view(card, Slot::RouterLogits)?,
                ids: view(card, Slot::TopkId)?,
                weights: view(card, Slot::TopkWeight)?,
                shared_gate: view(card, Slot::SharedGate)?,
                status: statuses[card],
            };
            let expert = ExpertBufs {
                packs: Packs::Q8 {
                    moe: view(card, Slot::Q8Moe)?,
                    down: view(card, Slot::Q8Down)?,
                    shared: view(card, Slot::Q8Shared)?,
                },
                expert_gu: view(card, Slot::ExpertGu)?,
                expert_down: view(card, Slot::ExpertDown)?,
                shared_gu: view(card, Slot::SharedGu)?,
                shared_out: view(card, Slot::SharedOut)?,
            };
            sets.push(MoeBuffers { route, expert });
        }
        Ok(sets)
    };

    // step_begin everywhere.
    for card in 0..cards {
        let params = stack.shards()[card].state().word(Word::StepParams)?;
        stack.contexts()[card].step_begin(&stack.streams()[card], params)?;
    }
    // The embedding: card 0 embeds, the row rides one broadcast.
    {
        let card0 = 0;
        let ids = stack.shards()[card0]
            .state()
            .step_field(StepField::TokenId)?;
        stack.contexts()[card0].embed_native(
            &stack.streams()[card0],
            EmbedNativeCall {
                weight: embed,
                ids,
                x: view(card0, Slot::ResidA)?,
                status: statuses[card0],
                m: 1,
            },
        )?;
        let outs: Vec<DeviceRange<'_>> = (0..cards)
            .map(|card| view(card, Slot::ResidA))
            .collect::<Result<_, _>>()?;
        stack.broadcast(0, view(0, Slot::ResidA)?, &outs, &ring_statuses, hidden)?;
    }
    // The layers: mixer block, then MoE block, residual flowing by parity.
    for (layer, &kind) in kinds.iter().enumerate() {
        let _ = kind;
        let resid: Vec<DeviceRange<'_>> = (0..cards)
            .map(|card| view(card, resid_in(layer)))
            .collect::<Result<_, _>>()?;
        run_card_block(
            stack,
            mixers,
            layer,
            ordinals[layer],
            block_ops[layer],
            &resid,
            admit_pos,
            kv_runs,
        )?;
        let sets = moe_bufs(layer)?;
        let lists: Vec<Vec<CardMoeOp>> = (0..cards)
            .map(|card| moe_sequence_card(card == 0))
            .collect();
        let ops: Vec<&[CardMoeOp]> = lists.iter().map(|l| l.as_slice()).collect();
        run_card_moe_block(stack, moe[layer], &sets, &ops)?;
    }
    // The tail on every card: final norm, the head shard, its argmax,
    // the advance.
    for card in 0..cards {
        let ctx = &stack.contexts()[card];
        let stream = &stack.streams()[card];
        let params = stack.shards()[card].state().word(Word::StepParams)?;
        let norm = output_norms[card].range(0, output_norms[card].bytes())?;
        ctx.add_rmsnorm_q8(
            stream,
            AddNormQ8Call {
                n: hidden,
                eps,
                h: view(card, final_resid(kinds.len()))?,
                delta: None,
                weight: norm,
                norm_out: view(card, Slot::FinalNormRow)?,
                q8: view(card, Slot::Q8Final)?,
            },
        )?;
        let head = &heads[card];
        let rows = head.n_rows();
        ctx.matvec_q(
            stream,
            MatvecCall {
                ty: head.quant(),
                w: head.buffer().range(0, head.buffer().bytes())?,
                x_q8: view(card, Slot::Q8Final)?,
                y: view(card, Slot::Logits)?,
                k: head.k(),
                rows,
                ncols: 1,
            },
        )?;
        ctx.argmax(
            stream,
            ArgmaxCall {
                n: rows,
                logits: view(card, Slot::Logits)?,
                part: view(card, Slot::ArgmaxPart)?,
                params,
                result: Some(stack.shards()[card].state().word(Word::ResultPair)?),
            },
        )?;
        ctx.step_advance(stream, params, 1)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::merge_head_candidates;

    #[test]
    fn the_merge_is_the_single_device_argmax_over_the_concatenated_shards() {
        // Ties across shards keep the LOWEST global id.
        let a = [1.0, 5.0, 2.0];
        let b = [5.0, 0.5, 5.0];
        assert_eq!(merge_head_candidates(&[&a, &b], &[0, 3]), 1);
        // A strictly larger value in the second shard wins by global id.
        let a = [1.0, 2.0];
        let b = [9.0, 0.0];
        assert_eq!(merge_head_candidates(&[&a, &b], &[0, 2]), 2);
        // NaN never wins; -inf is comparable and can win an all -inf row.
        let a = [f32::NAN, f32::NEG_INFINITY];
        let b = [f32::NAN, 1.0];
        assert_eq!(merge_head_candidates(&[&a, &b], &[0, 2]), 3);
        let all = [f32::NAN, f32::NAN];
        assert_eq!(merge_head_candidates(&[&all, &all], &[0, 2]), -1);
        let ninf = [f32::NEG_INFINITY, f32::NEG_INFINITY];
        assert_eq!(merge_head_candidates(&[&ninf, &ninf], &[0, 2]), 0);
    }
}
