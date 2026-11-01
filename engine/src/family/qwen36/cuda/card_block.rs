//! The card block driver: one mixer block of one layer across every card.
//!
//! This is the first executing piece of the distributed step. It walks ONE
//! card op list ([`block_ops_card`]; every card holds the identical
//! marker sequence) and routes each op:
//!
//! * [`CardOp::Local`] queues on the card's own ONE stream through the
//!   production `Mixers::run_local_op` - the same launches the
//!   single-device step uses, over the card's geometry, weight table and
//!   arena views;
//! * [`CardOp::GatherK`] / [`CardOp::GatherV`] run [`CardStack::
//!   all_gather_row`]: each card publishes the slice of the row its
//!   `kv_run` names, and every card's `KRows`/`VRows` slot ends with the
//!   full gathered row;
//! * [`CardOp::MixerSum`] runs [`CardStack::exchange_sum`] over the
//!   full-width `MixOut` partials in place - each card's out projection
//!   wrote its q-heads' contribution, and the summed row is the epilogue's
//!   mixer output.
//!
//! The v1 failure model of `multi.rs` is enforced here: after the gather
//! pair, the driver gates on both cards' ring status words BEFORE the
//! append queues (a dead gather must not corrupt the replicated cache);
//! every other exchange's status folds at the token's end.
//!
//! The ops run in list order, every card's work on its own stream, so the
//! one-stream-per-card invariant supplies the execution dependencies: a
//! card's publish sits after its producers in its stream, and a wait
//! cannot sum a slot before the peer's publish lands (the mailbox
//! sequence handshake).

use crate::EngError;
use crate::family::qwen36::cuda::card_step::CardStack;
use crate::family::qwen36::cuda::mixer::{BlockIo, Mixers};
use crate::family::qwen36::cuda::mixer_ops::CardOp;
use crate::family::qwen36::cuda::multi::fold_ring_status;
use crate::family::qwen36::cuda::step_plan::Slot;
use crate::kernels::cuda::DeviceRange;

const F32: usize = size_of::<f32>();

/// The marker of a card op: the ring markers by kind, every `Local` op as
/// `Local`. The per-card op lists must share one marker sequence - the
/// coordinator interleaves by position - while their `Local` payloads stay
/// per-card (the prep's absolute K/V row range above all).
fn marker(op: &CardOp) -> &'static str {
    match op {
        CardOp::Local(_) => "Local",
        CardOp::GatherK => "GatherK",
        CardOp::GatherV => "GatherV",
        CardOp::MixerSum => "MixerSum",
    }
}

/// Runs the block of `layer` on every card of `stack`. `ops[card]` is that
/// card's own op list (`CardKindOps`), one per card and all sharing the
/// same marker sequence; `mixers[card]` holds that card's parts, `ordinal`
/// the layer's state ordinal (the layer table's; the full-size per-card
/// states address their KV and delta rows at the single-device offsets),
/// `resid[card]` the view of the card's residual row (the op lists never
/// name the resid slots: the residual enters through `BlockIo`),
/// `admit_pos` the token position, and `kv_runs[card]` the card's absolute
/// K/V row range within the full row.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_card_block(
    stack: &CardStack,
    mixers: &[Mixers<'_>],
    layer: usize,
    ordinal: usize,
    ops: &[&[CardOp]],
    resid: &[DeviceRange<'_>],
    admit_pos: usize,
    kv_runs: &[std::ops::Range<usize>],
) -> Result<(), EngError> {
    let cards = stack.shards().len();
    if mixers.len() != cards || kv_runs.len() != cards || resid.len() != cards || ops.len() != cards
    {
        return Err(EngError::BadRequest(format!(
            "card block: {cards} cards, {} mixers, {} resid views, {} kv runs, {} op lists",
            mixers.len(),
            resid.len(),
            kv_runs.len(),
            ops.len()
        )));
    }
    for (card, list) in ops.iter().enumerate() {
        if list.len() != ops[0].len() {
            return Err(EngError::BadRequest(format!(
                "card block: card {card} has {} ops, card 0 has {}",
                list.len(),
                ops[0].len()
            )));
        }
        for (i, op) in list.iter().enumerate() {
            if marker(op) != marker(&ops[0][i]) {
                return Err(EngError::BadRequest(format!(
                    "card block: card {card} op {i} is {}, card 0 has {}",
                    marker(op),
                    marker(&ops[0][i])
                )));
            }
        }
    }
    for (i, op) in ops[0].iter().enumerate() {
        match op {
            CardOp::Local(_) => {
                for (card, mixer) in mixers.iter().enumerate() {
                    let CardOp::Local(block) = &ops[card][i] else {
                        unreachable!("the parity check pins Local markers by position");
                    };
                    mixer.run_local_op(
                        &stack.streams()[card],
                        stack.shards()[card].state(),
                        layer,
                        ordinal,
                        BlockIo {
                            resid: resid[card],
                            admit_pos,
                        },
                        block,
                    )?;
                }
            }
            CardOp::GatherK | CardOp::GatherV => {
                let slot = match op {
                    CardOp::GatherK => Slot::KRows,
                    _ => Slot::VRows,
                };
                let total = kv_runs.iter().map(|run| run.len()).sum::<usize>();
                let mut parts = Vec::with_capacity(cards);
                let mut outs = Vec::with_capacity(cards);
                let mut statuses = Vec::with_capacity(cards);
                for (card, mixer) in mixers.iter().enumerate() {
                    let row = mixer.slot_view(slot)?;
                    parts.push(row.buffer().range(
                        row.offset() + kv_runs[card].start * F32,
                        kv_runs[card].len() * F32,
                    )?);
                    outs.push(row.buffer().range(row.offset(), total * F32)?);
                    statuses.push(stack.shards()[card].ring_status()?);
                }
                stack.all_gather_row(&parts, &outs, &statuses, kv_runs, total)?;
                if matches!(op, CardOp::GatherV) {
                    // The gather gate of the v1 failure model: both halves
                    // published before the append may touch the cache.
                    gate_the_gathers(stack)?;
                }
            }
            CardOp::MixerSum => {
                let hidden = mixers[0].slot_view(Slot::MixOut)?.bytes() / F32;
                let mut partials = Vec::with_capacity(cards);
                let mut outs = Vec::with_capacity(cards);
                let mut statuses = Vec::with_capacity(cards);
                for (card, mixer) in mixers.iter().enumerate() {
                    let row = mixer.slot_view(Slot::MixOut)?;
                    partials.push(row.buffer().range(row.offset(), row.bytes())?);
                    outs.push(row.buffer().range(row.offset(), row.bytes())?);
                    statuses.push(stack.shards()[card].ring_status()?);
                }
                stack.exchange_sum(
                    &partials,
                    &outs,
                    &statuses,
                    hidden,
                    crate::kernels::cuda::peer_ring::RingMode::F32,
                )?;
            }
        }
    }
    Ok(())
}

/// Reads every card's ring status word and refuses - session-fatal, the
/// rig's rule - when any gather timed out or overflowed. The refusal
/// happens BEFORE the append queues, so a dead gather cannot write a
/// stale row into the replicated cache.
fn gate_the_gathers(stack: &CardStack) -> Result<(), EngError> {
    let mut words = Vec::with_capacity(stack.shards().len());
    for (card, shard) in stack.shards().iter().enumerate() {
        stack.streams()[card].synchronize()?;
        let mut bytes = [0u8; 4];
        shard
            .ring_word()
            .read_at(&stack.streams()[card], 0, &mut bytes)?;
        words.push(u32::from_le_bytes(bytes));
    }
    let folded = fold_ring_status(&words);
    if folded != 0 {
        return Err(EngError::BadRequest(format!(
            "card block: the K/V gather failed on the ring (status {folded}); \
             the session is dead and the cache was not appended"
        )));
    }
    Ok(())
}
