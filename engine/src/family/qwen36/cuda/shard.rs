//! How one Qwen3.6 layer is divided across CUDA devices: the pure range
//! arithmetic of the multi-device step.
//!
//! One [`ShardTable`] is built from the layout record ([`HostShape`]), a card
//! count, and the K-alignment the column-sharded weights need. It owns every
//! range a loader or a step shard may ask for, so no other module deals heads
//! or widths by hand. Ranges are dealt in contiguous runs; when a count
//! does not divide, the earlier cards take one extra unit. Full attention
//! is the exception: its query heads move in whole 8-head groups (the
//! decode kernel's `n_head == 8 * n_kv_head` grouping), so 16 query heads
//! on 3 cards refuse rather than deal 6/5/5. A table for one card is the full range everywhere,
//! which is what keeps the single-device path the degenerate case of this
//! arithmetic rather than a second code path.
//!
//! The sharding units, and why:
//! - DeltaNet is dealt by **key-head group**. Each key head owns
//!   `value_heads / key_heads` value heads (2 on Qwen3.6-35B-A3B, a checked
//!   ratio of `HostShape`), and a run never splits a key head's value heads,
//!   so the recurrent state, the conv channels and the `out` columns of a
//!   group stay on one card.
//! - Full attention is dealt by **query-head group**. The shard unit of
//!   `AttnQ` is one head's fused query-then-gate rows, but the decode
//!   kernel serves exactly eight query heads per K/V head, and a card
//!   decodes its query heads against its own K/V slice of the replicated
//!   row, so runs keep whole 8-head groups: 16 query heads on 2 cards deal
//!   8/8, each card owning the K/V head its group reads (the step
//!   all-gathers each new K/V row, so the cache stays replicated).
//! - Experts are dealt by **intermediate width inside every expert**, in
//!   32-value blocks, so every card holds a slice of every expert and routing
//!   never moves activations between cards.
//! - The output head is dealt by **vocabulary rows**.
//!
//! Column-sharded matrices (`o`, DeltaNet `out`, expert `down`) produce a
//! partial hidden row per card. Their column cuts must land on quantization
//! block boundaries along K: a GGUF block cannot be split between cards.
//! The caller passes that alignment (`k_align`, 256 for the K-quant
//! superblocks); a table whose cut would split a block is refused here, by
//! name, before any tensor is read. On Qwen3.6 the expert `down` K width is
//! 512, so 3 cards at alignment 256 refuse — that refusal is the table doing
//! its job, and the loader decides per tensor type what alignment applies.

use std::ops::Range;

use super::super::host::HostShape;
use crate::EngError;
use crate::family::gguf_model::{Inventory, Role};
use crate::family::project::cuda::SliceSpec;

/// Values per dealt block of an expert's intermediate width and of a K/V row.
pub(crate) const VALUE_BLOCK: usize = 32;

/// The decode kernel serves exactly eight query heads per K/V head
/// (`n_head == 8 * n_kv_head`, resident/attn_decode.h), and a card
/// decodes its dealt query heads against its OWN K/V slice of the
/// replicated row (`card_prep_shape`). Legal runs therefore keep whole
/// 8-head query groups, each card's K/V rows following its groups.
pub(crate) const ATTN_Q_PER_KV_HEAD: usize = 8;

fn refuse(message: String) -> EngError {
    EngError::BadRequest(format!("shard table: {message}"))
}

/// Deals `total` units of `block` values each over `cards` contiguous runs.
/// Runs differ by at most one block; the earlier cards take the extra block.
/// Refuses a total that is not a whole number of blocks, and a card count
/// that would leave a card with no block when blocks run out.
fn deal(
    total: usize,
    block: usize,
    cards: usize,
    what: &str,
) -> Result<Vec<Range<usize>>, EngError> {
    if block == 0 || !total.is_multiple_of(block) {
        return Err(refuse(format!(
            "{what} width {total} is not a whole number of {block}-value blocks"
        )));
    }
    let blocks = total / block;
    if cards == 0 || cards > blocks {
        return Err(refuse(format!(
            "{what} has {blocks} blocks for {cards} cards; a card would hold none"
        )));
    }
    let (base, extra) = (blocks / cards, blocks % cards);
    let mut runs = Vec::with_capacity(cards);
    let mut start = 0;
    for card in 0..cards {
        let len = (base + usize::from(card < extra)) * block;
        runs.push(start..start + len);
        start += len;
    }
    Ok(runs)
}

/// Refuses when a boundary inside `runs` (an end that is not the total) is not
/// a multiple of `k_align`: that cut would split a quantization block.
fn check_k_cuts(runs: &[Range<usize>], k_align: usize, what: &str) -> Result<(), EngError> {
    if k_align <= 1 {
        return Ok(());
    }
    for run in &runs[..runs.len().saturating_sub(1)] {
        if !run.end.is_multiple_of(k_align) {
            return Err(refuse(format!(
                "{what} column cut at {} splits a {k_align}-value quantization block",
                run.end
            )));
        }
    }
    Ok(())
}

/// Every per-card range of one Qwen3.6 layer, plus the head and embedding
/// sides. Built once at engine start; read by the loader and the step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ShardTable {
    cards: usize,
    key_heads: Vec<Range<usize>>,
    value_heads: Vec<Range<usize>>,
    query_heads: Vec<Range<usize>>,
    kv_rows: Vec<Range<usize>>,
    inter: Vec<Range<usize>>,
    shared_inter: Vec<Range<usize>>,
    vocab: Vec<Range<usize>>,
    /// Value heads per key head of the shape the table was built from.
    key_dim: usize,
    value_dim: usize,
    head_dim: usize,
    total_key_heads: usize,
}

/// Explicit per-card runs for every dealt dimension, as produced by the
/// placement planner (`placement`). `from_runs` enforces exactly what
/// `new` guarantees: contiguous whole-grain coverage and `k_align`
/// column cuts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ShardRuns {
    pub(crate) value_heads: Vec<Range<usize>>,
    pub(crate) query_heads: Vec<Range<usize>>,
    pub(crate) kv_rows: Vec<Range<usize>>,
    pub(crate) inter: Vec<Range<usize>>,
    pub(crate) shared_inter: Vec<Range<usize>>,
    pub(crate) vocab: Vec<Range<usize>>,
}

/// Refuses runs that are not a contiguous full covering of `total` in
/// whole `grain` units, or that leave a card with an empty run.
fn check_runs(
    runs: &[Range<usize>],
    total: usize,
    grain: usize,
    what: &str,
) -> Result<(), EngError> {
    let mut at = 0;
    for (card, run) in runs.iter().enumerate() {
        if run.is_empty() {
            return Err(refuse(format!("{what} run {card} is empty")));
        }
        if !run.len().is_multiple_of(grain) {
            return Err(refuse(format!(
                "{what} run {card} of {} values splits {grain}-value blocks",
                run.len()
            )));
        }
        if run.start != at {
            return Err(refuse(format!(
                "{what} runs are not contiguous at card {card}"
            )));
        }
        at = run.end;
    }
    if at != total {
        return Err(refuse(format!("{what} runs cover {at} of {total}")));
    }
    Ok(())
}

impl ShardTable {
    /// Builds the table for `cards` devices. `k_align` is the quantization
    /// block width along K that the column-sharded matrices (`o`, `out`,
    /// `down`) must not split; pass 1 when the weight layout has no such
    /// block (row tests and host planning).
    pub(crate) fn new(shape: &HostShape, cards: usize, k_align: usize) -> Result<Self, EngError> {
        shape.check()?;
        if cards == 0 {
            return Err(refuse("a shard table needs at least one card".to_owned()));
        }
        let (d, a, m) = (&shape.delta, &shape.attn, &shape.moe);
        if a.heads_q != ATTN_Q_PER_KV_HEAD * a.heads_kv {
            return Err(refuse(format!(
                "attention geometry {} query heads over {} K/V heads is not the decode \
                 kernel's 8-per-head grouping; per-card decode cannot run it",
                a.heads_q, a.heads_kv
            )));
        }
        // The delta kernel pairs value head h with key head h % TOTAL key
        // heads, so a card's value heads need keys from across the whole
        // model whenever the value/key ratio exceeds 1. Keys therefore
        // REPLICATE: every card holds all key channels, and only the value
        // heads are dealt.
        let key_heads = vec![0..d.key_heads; cards];
        let value_heads = deal(d.value_heads, 1, cards, "DeltaNet value heads")?;
        let query_heads = deal(
            a.heads_q,
            ATTN_Q_PER_KV_HEAD,
            cards,
            "attention query heads",
        )?;
        // K/V rows follow the query groups: the card holding group g
        // stores, and decodes against, K/V head g's rows.
        let kv_rows = query_heads
            .iter()
            .map(|q| {
                q.start / ATTN_Q_PER_KV_HEAD * a.head_dim..q.end / ATTN_Q_PER_KV_HEAD * a.head_dim
            })
            .collect();
        let inter = deal(m.inter, VALUE_BLOCK, cards, "expert intermediate")?;
        let shared_inter = deal(m.shared_inter, VALUE_BLOCK, cards, "shared intermediate")?;
        let vocab = deal(shape.vocab, 1, cards, "vocabulary")?;
        // Column cuts of the three partial-sum matrices, in K values.
        let delta_out: Vec<Range<usize>> = value_heads
            .iter()
            .map(|run| run.start * d.value_dim..run.end * d.value_dim)
            .collect();
        check_k_cuts(&delta_out, k_align, "DeltaNet out")?;
        let attn_o: Vec<Range<usize>> = query_heads
            .iter()
            .map(|run| run.start * a.head_dim..run.end * a.head_dim)
            .collect();
        check_k_cuts(&attn_o, k_align, "attention o")?;
        check_k_cuts(&inter, k_align, "expert down")?;
        check_k_cuts(&shared_inter, k_align, "shared down")?;
        Ok(Self {
            cards,
            key_heads,
            value_heads,
            query_heads,
            kv_rows,
            inter,
            shared_inter,
            vocab,
            key_dim: d.key_dim,
            value_dim: d.value_dim,
            head_dim: a.head_dim,
            total_key_heads: d.key_heads,
        })
    }

    /// Builds the table from planner-chosen runs instead of the default
    /// even deal. Same validation as `new`: whole-grain contiguous full
    /// coverage per dimension, and `k_align` column cuts on the
    /// partial-sum matrices.
    pub(crate) fn from_runs(
        shape: &HostShape,
        k_align: usize,
        runs: &ShardRuns,
    ) -> Result<Self, EngError> {
        shape.check()?;
        let cards = runs.value_heads.len();
        if cards == 0 {
            return Err(refuse("a shard table needs at least one card".to_owned()));
        }
        let specs = [
            (
                runs.value_heads.as_slice(),
                shape.delta.value_heads,
                1,
                "DeltaNet value heads",
            ),
            (
                runs.query_heads.as_slice(),
                shape.attn.heads_q,
                ATTN_Q_PER_KV_HEAD,
                "attention query heads",
            ),
            (
                runs.kv_rows.as_slice(),
                shape.attn.heads_kv * shape.attn.head_dim,
                VALUE_BLOCK,
                "K/V rows",
            ),
            (
                runs.inter.as_slice(),
                shape.moe.inter,
                VALUE_BLOCK,
                "expert intermediate",
            ),
            (
                runs.shared_inter.as_slice(),
                shape.moe.shared_inter,
                VALUE_BLOCK,
                "shared intermediate",
            ),
            (runs.vocab.as_slice(), shape.vocab, 1, "vocabulary"),
        ];
        for (dim_runs, total, grain, what) in specs {
            if dim_runs.len() != cards {
                return Err(refuse(format!(
                    "{what} has {} runs for {cards} cards",
                    dim_runs.len()
                )));
            }
            check_runs(dim_runs, total, grain, what)?;
        }
        let delta_out: Vec<Range<usize>> = runs
            .value_heads
            .iter()
            .map(|run| run.start * shape.delta.value_dim..run.end * shape.delta.value_dim)
            .collect();
        check_k_cuts(&delta_out, k_align, "DeltaNet out")?;
        let attn_o: Vec<Range<usize>> = runs
            .query_heads
            .iter()
            .map(|run| run.start * shape.attn.head_dim..run.end * shape.attn.head_dim)
            .collect();
        check_k_cuts(&attn_o, k_align, "attention o")?;
        check_k_cuts(&runs.inter, k_align, "expert down")?;
        check_k_cuts(&runs.shared_inter, k_align, "shared down")?;
        if shape.attn.heads_q != ATTN_Q_PER_KV_HEAD * shape.attn.heads_kv {
            return Err(refuse(format!(
                "attention geometry {} query heads over {} K/V heads is not the decode \
                 kernel's 8-per-head grouping",
                shape.attn.heads_q, shape.attn.heads_kv
            )));
        }
        for card in 0..cards {
            let q = &runs.query_heads[card];
            let kv = &runs.kv_rows[card];
            if !kv.len().is_multiple_of(shape.attn.head_dim) {
                return Err(refuse(format!(
                    "card {card} holds {} K/V rows, which split a {}-row head",
                    kv.len(),
                    shape.attn.head_dim
                )));
            }
            let kv_heads_card = kv.len() / shape.attn.head_dim;
            if q.len() != ATTN_Q_PER_KV_HEAD * kv_heads_card
                || q.start != kv.start / shape.attn.head_dim * ATTN_Q_PER_KV_HEAD
            {
                return Err(refuse(format!(
                    "card {card} holds {} query heads over {} K/V heads; the decode kernel \
                     groups 8 query heads per K/V head, so runs must keep whole groups",
                    q.len(),
                    kv_heads_card
                )));
            }
        }
        Ok(Self {
            cards,
            key_heads: vec![0..shape.delta.key_heads; cards],
            value_heads: runs.value_heads.clone(),
            query_heads: runs.query_heads.clone(),
            kv_rows: runs.kv_rows.clone(),
            inter: runs.inter.clone(),
            shared_inter: runs.shared_inter.clone(),
            vocab: runs.vocab.clone(),
            key_dim: shape.delta.key_dim,
            value_dim: shape.delta.value_dim,
            head_dim: shape.attn.head_dim,
            total_key_heads: shape.delta.key_heads,
        })
    }

    pub(crate) fn cards(&self) -> usize {
        self.cards
    }

    fn at<'a>(
        &'a self,
        runs: &'a [Range<usize>],
        card: usize,
        what: &str,
    ) -> Result<Range<usize>, EngError> {
        runs.get(card).cloned().ok_or_else(|| {
            refuse(format!(
                "{what} has no run for card {card} of {}",
                self.cards
            ))
        })
    }

    /// The key heads `card` runs in every DeltaNet layer: the FULL set on
    /// every card (keys replicate; see `new`), so a card's value heads
    /// always find their `h % total` key channels locally.
    pub(crate) fn key_heads(&self, card: usize) -> Result<Range<usize>, EngError> {
        self.at(&self.key_heads, card, "DeltaNet key heads")
    }

    /// The value heads `card` runs: the dealt contiguous run.
    pub(crate) fn value_heads(&self, card: usize) -> Result<Range<usize>, EngError> {
        self.at(&self.value_heads, card, "DeltaNet value heads")
    }

    /// The three channel segments of the fused DeltaNet `qkv` projection that
    /// `card` holds, in the stored channel order: the query channels and key
    /// channels (REPLICATED in full on every card) and its dealt value
    /// channels, each as a range inside `conv_dim` (`[key heads | key heads |
    /// value heads]` channels).
    pub(crate) fn delta_qkv_segments(&self, card: usize) -> Result<[Range<usize>; 3], EngError> {
        let values = self.value_heads(card)?;
        let key_width = self.total_key_heads * self.key_dim;
        Ok([
            0..key_width,
            key_width..2 * key_width,
            2 * key_width + values.start * self.value_dim
                ..2 * key_width + values.end * self.value_dim,
        ])
    }

    /// The output rows of the DeltaNet gate projection `z` of `card` (one
    /// value-head row block per value head).
    pub(crate) fn delta_z_rows(&self, card: usize) -> Result<Range<usize>, EngError> {
        let values = self.value_heads(card)?;
        Ok(values.start * self.value_dim..values.end * self.value_dim)
    }

    /// The columns of the DeltaNet `out` projection `card` multiplies; its
    /// result is a partial hidden row.
    pub(crate) fn delta_out_cols(&self, card: usize) -> Result<Range<usize>, EngError> {
        self.delta_z_rows(card)
    }

    /// The query heads `card` runs in every full-attention layer.
    pub(crate) fn query_heads(&self, card: usize) -> Result<Range<usize>, EngError> {
        self.at(&self.query_heads, card, "attention query heads")
    }

    /// The output rows of the fused `AttnQ` projection of `card`: per head,
    /// its query rows then its gate rows, so one head is `2 * head_dim` rows.
    pub(crate) fn attn_q_rows(&self, card: usize) -> Result<Range<usize>, EngError> {
        let run = self.query_heads(card)?;
        Ok(run.start * 2 * self.head_dim..run.end * 2 * self.head_dim)
    }

    /// The rows of each of `AttnK` and `AttnV` that `card` computes before the
    /// row is all-gathered into every card's replicated KV cache.
    pub(crate) fn kv_rows(&self, card: usize) -> Result<Range<usize>, EngError> {
        self.at(&self.kv_rows, card, "K/V rows")
    }

    /// The columns of the attention `o` projection `card` multiplies.
    pub(crate) fn attn_o_cols(&self, card: usize) -> Result<Range<usize>, EngError> {
        let run = self.query_heads(card)?;
        Ok(run.start * self.head_dim..run.end * self.head_dim)
    }

    /// The intermediate values of every routed expert that `card` computes;
    /// the same range cuts its `gate`/`up` rows and its `down` columns.
    pub(crate) fn expert_inter(&self, card: usize) -> Result<Range<usize>, EngError> {
        self.at(&self.inter, card, "expert intermediate")
    }

    /// The intermediate values of the shared expert that `card` computes.
    pub(crate) fn shared_inter(&self, card: usize) -> Result<Range<usize>, EngError> {
        self.at(&self.shared_inter, card, "shared intermediate")
    }

    /// The vocabulary rows of the output head that `card` computes; the token
    /// id of a local row is the range start plus the local index.
    pub(crate) fn head_rows(&self, card: usize) -> Result<Range<usize>, EngError> {
        self.at(&self.vocab, card, "vocabulary")
    }
}

/// What one card loads of one inventory tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ShardLoad {
    /// The whole tensor, replicated (norms, router, per-head vectors).
    Whole,
    /// Only card 0 loads it (the embedding table; its row is broadcast).
    CardZeroOnly,
    /// This card's slice, in GGUF axis order.
    Sliced(SliceSpec),
}

/// One row range as a range list, for the roles sliced along one run.
fn one(range: Range<u64>) -> Vec<Range<u64>> {
    std::iter::once(range).collect()
}

fn as_u64(range: Range<usize>) -> Range<u64> {
    range.start as u64..range.end as u64
}

/// The load plan of one card over the whole inventory: every tensor the
/// model holds, classified whole, card-0-only, or sliced by the table's
/// ranges for its role. GGUF axes are `dims[0]` = row length (the input
/// width a column-sharded matrix is cut along) and `dims[1]` = rows (the
/// output width a row-sharded matrix is cut along), with expert stacks
/// cut inside every plane. The union of all cards' slices of a tensor is
/// the tensor, and the 1-card plan is `Whole` everywhere except the
/// embedding, which its one card owns either way.
pub(crate) fn shard_loads(
    table: &ShardTable,
    inventory: &Inventory,
    card: usize,
) -> Result<Vec<(String, ShardLoad)>, EngError> {
    if card >= table.cards() {
        return Err(refuse(format!(
            "card {card} is outside the {}-card table",
            table.cards()
        )));
    }
    // One card loads everything whole (and through plain `load` at start);
    // the slices below are its full ranges anyway.
    let sliced = |cols: Range<u64>, rows: Vec<Range<u64>>| {
        if table.cards() == 1 {
            ShardLoad::Whole
        } else {
            ShardLoad::Sliced(SliceSpec { cols, rows })
        }
    };
    inventory
        .tensors()
        .iter()
        .map(|tensor| {
            let dims = &tensor.dims;
            let full_cols = 0..dims[0];
            let full_rows = one(0..dims.get(1).copied().unwrap_or(1));
            let load = match tensor.role {
                Role::TokenEmbd => ShardLoad::CardZeroOnly,
                Role::Output => sliced(full_cols, one(as_u64(table.head_rows(card)?))),
                Role::AttnQ => sliced(full_cols, one(as_u64(table.attn_q_rows(card)?))),
                Role::AttnK | Role::AttnV => sliced(full_cols, one(as_u64(table.kv_rows(card)?))),
                Role::AttnOutput => sliced(as_u64(table.attn_o_cols(card)?), full_rows),
                Role::AttnQkv | Role::SsmConv1d => sliced(
                    full_cols,
                    table
                        .delta_qkv_segments(card)?
                        .into_iter()
                        .map(as_u64)
                        .collect(),
                ),
                Role::AttnGate => sliced(full_cols, one(as_u64(table.delta_z_rows(card)?))),
                Role::SsmOut => sliced(as_u64(table.delta_out_cols(card)?), full_rows),
                Role::SsmAlpha | Role::SsmBeta => {
                    sliced(full_cols, one(as_u64(table.value_heads(card)?)))
                }
                Role::SsmA | Role::SsmDtBias => sliced(as_u64(table.value_heads(card)?), one(0..1)),
                Role::FfnGateExps | Role::FfnUpExps => {
                    sliced(full_cols, one(as_u64(table.expert_inter(card)?)))
                }
                Role::FfnDownExps => sliced(as_u64(table.expert_inter(card)?), full_rows),
                Role::FfnGateShexp | Role::FfnUpShexp => {
                    sliced(full_cols, one(as_u64(table.shared_inter(card)?)))
                }
                Role::FfnDownShexp => sliced(as_u64(table.shared_inter(card)?), full_rows),
                // Replicated: norms and per-head vectors every card applies to
                // its own heads, the router (computed on card 0 in the step;
                // loading it everywhere keeps the load plan shape-free), and
                // the shared-expert gate scalar inputs.
                Role::OutputNorm
                | Role::AttnNorm
                | Role::PostAttentionNorm
                | Role::FfnGateInp
                | Role::FfnGateInpShexp
                | Role::AttnQNorm
                | Role::AttnKNorm
                | Role::SsmNorm => ShardLoad::Whole,
            };
            Ok((tensor.name.clone(), load))
        })
        .collect()
}

/// One card's greedy candidate from its vocabulary shard: the global token
/// id and its logit, as the existing device argmax produces them per shard.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct HeadCandidate {
    pub(crate) token: u32,
    pub(crate) logit: f32,
}

/// The winner over per-card candidates, by the one argmax rule of the
/// engine (`family::sampler` and the device `beats()` in
/// `native/resident/step_params.cu`): the largest value, the lowest token
/// id on a tie, and a NaN never wins. Candidates carry global token ids,
/// so the rule needs no shard knowledge; an empty or all-NaN set has no
/// winner, matching the device kernel's no-candidate result. Because the
/// rule is a strict total order on (value, id) pairs, the winner equals
/// the unsplit argmax over the same logits.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "staged for the two-device step, which merges per-shard candidates"
    )
)]
pub(crate) fn merge_head_candidates(candidates: &[HeadCandidate]) -> Option<HeadCandidate> {
    let mut best: Option<HeadCandidate> = None;
    for &candidate in candidates {
        let wins = match best {
            None => !candidate.logit.is_nan(),
            Some(current) => {
                candidate.logit > current.logit
                    || (candidate.logit == current.logit && candidate.token < current.token)
            }
        };
        if wins {
            best = Some(candidate);
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::spec::QWEN3_6_35B_A3B;

    fn shape() -> HostShape {
        HostShape::from_spec(&QWEN3_6_35B_A3B, 64).expect("real shape")
    }

    /// The runs of one accessor over all cards, in card order.
    fn runs(
        table: &ShardTable,
        get: impl Fn(&ShardTable, usize) -> Result<Range<usize>, EngError>,
    ) -> Vec<Range<usize>> {
        (0..table.cards())
            .map(|card| get(table, card).expect("run"))
            .collect()
    }

    fn assert_partition(runs: &[Range<usize>], total: usize, what: &str) {
        let mut next = 0;
        for run in runs {
            assert_eq!(run.start, next, "{what}: gap or overlap before {run:?}");
            assert!(run.end > run.start, "{what}: empty run {run:?}");
            next = run.end;
        }
        assert_eq!(next, total, "{what}: runs do not cover the width");
    }

    #[test]
    fn every_width_partitions_for_one_and_two_cards() {
        let shape = shape();
        for cards in 1..=2 {
            let table = ShardTable::new(&shape, cards, 32).expect("table");
            // Keys replicate (the delta kernel pairs value head h with key
            // head h % 16, so every card needs all of them); values deal.
            for card in 0..cards {
                assert_eq!(table.key_heads(card).expect("run"), 0..16);
            }
            assert_partition(&runs(&table, ShardTable::value_heads), 32, "value heads");
            assert_partition(&runs(&table, ShardTable::query_heads), 16, "query heads");
            assert_partition(&runs(&table, ShardTable::attn_q_rows), 8192, "attn q rows");
            assert_partition(&runs(&table, ShardTable::kv_rows), 512, "kv rows");
            assert_partition(&runs(&table, ShardTable::attn_o_cols), 4096, "attn o cols");
            assert_partition(
                &runs(&table, ShardTable::delta_out_cols),
                4096,
                "delta out cols",
            );
            assert_partition(&runs(&table, ShardTable::expert_inter), 512, "expert inter");
            assert_partition(&runs(&table, ShardTable::shared_inter), 512, "shared inter");
            assert_partition(&runs(&table, ShardTable::head_rows), 248_320, "vocab");
        }
        // The 16 query heads are two whole 8-head decode groups; three
        // and four cards cannot each own one, so the table refuses by
        // name instead of dealing heads the kernel cannot run.
        for cards in 3..=4 {
            match ShardTable::new(&shape, cards, 32) {
                Err(EngError::BadRequest(message)) => {
                    assert!(message.contains("attention query heads"), "{message}");
                }
                other => panic!("{cards} cards must refuse, got {other:?}"),
            }
        }
    }

    #[test]
    fn one_card_is_the_full_range_everywhere() {
        let table = ShardTable::new(&shape(), 1, 256).expect("one card");
        assert_eq!(table.key_heads(0).expect("run"), 0..16);
        assert_eq!(table.expert_inter(0).expect("run"), 0..512);
        assert_eq!(table.head_rows(0).expect("run"), 0..248_320);
        assert_eq!(
            table.delta_qkv_segments(0).expect("segments"),
            [0..2048, 2048..4096, 4096..8192]
        );
    }

    #[test]
    fn two_cards_deal_the_documented_runs_and_three_refuse() {
        let table = ShardTable::new(&shape(), 2, 32).expect("two cards");
        assert_eq!(runs(&table, ShardTable::key_heads), vec![0..16, 0..16]);
        assert_eq!(runs(&table, ShardTable::value_heads), vec![0..16, 16..32]);
        // Query heads move in whole 8-head groups (the decode kernel
        // serves 8 query heads per K/V head); K/V rows follow the groups.
        assert_eq!(runs(&table, ShardTable::query_heads), vec![0..8, 8..16]);
        assert_eq!(runs(&table, ShardTable::kv_rows), vec![0..256, 256..512]);
        // 16 blocks of 32 dealt 8/8: the earlier card takes the extra block.
        assert_eq!(
            runs(&table, ShardTable::expert_inter),
            vec![0..256, 256..512]
        );
        // Card 1's qkv segments: replicated key channels (q, k in full)
        // and its dealt value channels 16..32.
        assert_eq!(
            table.delta_qkv_segments(1).expect("segments"),
            [0..2048, 2048..4096, 6144..8192]
        );
        // Three cards cannot each own one of the two query groups.
        let error = ShardTable::new(&shape(), 3, 32).expect_err("three cards must refuse");
        match error {
            EngError::BadRequest(message) => {
                assert!(message.contains("attention query heads"), "{message}");
            }
            other => panic!("want BadRequest, got {other:?}"),
        }
    }

    #[test]
    fn a_cut_inside_a_quantization_block_is_refused_by_name() {
        // A 17/15 value-head deal cuts the DeltaNet out columns at
        // 17 * 128 = 2176, mid-way through a 256-value superblock (value
        // heads deal at grain 1, so the cut may land mid-block). The
        // refusal names the first bad cut.
        let even = ShardTable::new(&shape(), 2, 256).expect("two cards at 256");
        let bad = ShardRuns {
            value_heads: vec![0..17, 17..32],
            query_heads: (0..2)
                .map(|c| even.query_heads(c).expect("q run"))
                .collect(),
            kv_rows: (0..2).map(|c| even.kv_rows(c).expect("kv run")).collect(),
            inter: (0..2)
                .map(|c| even.expert_inter(c).expect("inter run"))
                .collect(),
            shared_inter: (0..2)
                .map(|c| even.shared_inter(c).expect("shared run"))
                .collect(),
            vocab: (0..2)
                .map(|c| even.head_rows(c).expect("vocab run"))
                .collect(),
        };
        let error =
            ShardTable::from_runs(&shape(), 256, &bad).expect_err("a mid-block cut must refuse");
        match error {
            EngError::BadRequest(message) => {
                assert!(message.contains("DeltaNet out"), "{message}");
                assert!(message.contains("2176"), "{message}");
            }
            other => panic!("want BadRequest, got {other:?}"),
        }
    }

    fn candidate(token: u32, logit: f32) -> HeadCandidate {
        HeadCandidate { token, logit }
    }

    #[test]
    fn the_head_merge_follows_the_one_argmax_rule() {
        // Largest wins; on a tie the lowest global token id, in any order.
        let tied = [candidate(90, 3.0), candidate(12, 3.0), candidate(40, 2.5)];
        assert_eq!(merge_head_candidates(&tied), Some(candidate(12, 3.0)));
        let shuffled = [candidate(40, 2.5), candidate(90, 3.0), candidate(12, 3.0)];
        assert_eq!(merge_head_candidates(&shuffled), Some(candidate(12, 3.0)));
        // +infinity wins like a value; NaN never wins, even alone.
        assert_eq!(
            merge_head_candidates(&[candidate(1, f32::NAN), candidate(2, f32::INFINITY)]),
            Some(candidate(2, f32::INFINITY))
        );
        assert_eq!(merge_head_candidates(&[candidate(1, f32::NAN)]), None);
        assert_eq!(merge_head_candidates(&[]), None);
        // The merged winner over shard maxima equals the unsplit argmax:
        // split one row at the table's vocab ranges and compare.
        let table = ShardTable::new(&shape(), 2, 32).expect("table");
        let row: Vec<f32> = (0..248_320)
            .map(|i| ((i * 2_654_435_761usize) % 1000) as f32 * 0.001)
            .collect();
        let mut shards = Vec::new();
        for card in 0..2 {
            let range = table.head_rows(card).expect("vocab range");
            let (at, &logit) = row[range.clone()].iter().enumerate().fold(
                (0usize, &f32::NEG_INFINITY),
                |best, (i, v)| {
                    if v > best.1 { (i, v) } else { best }
                },
            );
            shards.push(candidate((range.start + at) as u32, logit));
        }
        let whole = row
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |best, (i, &v)| {
                if v > best.1 { (i, v) } else { best }
            });
        assert_eq!(
            merge_head_candidates(&shards),
            Some(candidate(whole.0 as u32, whole.1))
        );
    }

    #[test]
    fn two_card_load_plans_classify_and_cover_the_whole_inventory() {
        use crate::weights::GgufType;
        let inventory = Inventory::for_spec(&QWEN3_6_35B_A3B).expect("inventory");
        let table = ShardTable::new(&shape(), 2, 256).expect("two cards at 256");
        let plans = [
            shard_loads(&table, &inventory, 0).expect("card 0 plan"),
            shard_loads(&table, &inventory, 1).expect("card 1 plan"),
        ];
        assert_eq!(plans[0].len(), inventory.tensors().len());
        for (tensor, (name0, load0)) in inventory.tensors().iter().zip(&plans[0]) {
            assert_eq!(&tensor.name, name0);
            let load1 = &plans[1]
                .iter()
                .find(|(name, _)| name == name0)
                .expect("card 1 entry")
                .1;
            let ty = match tensor.path() {
                crate::family::gguf_model::TensorPath::DeviceTiled => GgufType::Q4K,
                crate::family::gguf_model::TensorPath::DeviceF32 => GgufType::F32,
            };
            let whole = crate::kernels::cuda::plan::native_device_bytes(name0, ty, &tensor.dims)
                .expect("whole bytes");
            let slice_bytes = |load: &ShardLoad| -> u64 {
                match load {
                    ShardLoad::Whole => whole,
                    ShardLoad::CardZeroOnly => whole,
                    ShardLoad::Sliced(spec) => {
                        crate::kernels::cuda::plan::plan_slice_rows(
                            name0,
                            ty,
                            &tensor.dims,
                            spec.cols.clone(),
                            &spec.rows,
                        )
                        .unwrap_or_else(|e| panic!("{name0} slice must plan: {e:?}"))
                        .bytes
                    }
                }
            };
            match (load0, load1) {
                (ShardLoad::CardZeroOnly, ShardLoad::CardZeroOnly) => {
                    assert_eq!(tensor.role, Role::TokenEmbd, "{name0}");
                }
                (ShardLoad::Whole, ShardLoad::Whole) => {}
                (ShardLoad::Sliced(spec0), ShardLoad::Sliced(_)) => {
                    // The fused DeltaNet QKV and its conv replicate the
                    // key channels on every card - `delta_qkv_segments` gives
                    // each card TWO full-key-width row blocks with only the
                    // third block's value rows dealt - so the pair over-covers
                    // the tensor by exactly one replicated copy; every other
                    // slice tiles the tensor exactly.
                    let key_channels = (2 * shape().delta.key_heads * shape().delta.key_dim) as u64;
                    let expected = match tensor.role {
                        Role::AttnQkv | Role::SsmConv1d => {
                            let key_rows = 0..key_channels;
                            let key_copy = crate::kernels::cuda::plan::plan_slice_rows(
                                name0,
                                ty,
                                &tensor.dims,
                                spec0.cols.clone(),
                                &[key_rows],
                            )
                            .expect("the replicated key slice must plan")
                            .bytes;
                            whole + key_copy
                        }
                        _ => whole,
                    };
                    assert_eq!(
                        slice_bytes(load0) + slice_bytes(load1),
                        expected,
                        "{name0}: the two slices must cover the tensor exactly"
                    );
                }
                other => panic!("{name0}: asymmetric load plan {other:?}"),
            }
        }
        assert!(shard_loads(&table, &inventory, 2).is_err());
        // The 1-card plan loads everything whole.
        let one = ShardTable::new(&shape(), 1, 256).expect("one card");
        for (_, load) in shard_loads(&one, &inventory, 0).expect("one-card plan") {
            assert!(matches!(load, ShardLoad::Whole | ShardLoad::CardZeroOnly));
        }
    }

    #[test]
    fn zero_cards_and_a_card_past_the_table_refuse() {
        assert!(ShardTable::new(&shape(), 0, 1).is_err());
        let table = ShardTable::new(&shape(), 2, 1).expect("two cards");
        assert!(table.key_heads(2).is_err());
        assert!(table.head_rows(9).is_err());
    }
}
