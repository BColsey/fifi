//! How one Qwen3.6 layer is divided across CUDA devices: the pure range
//! arithmetic of the multi-device step.
//!
//! One [`ShardTable`] is built from the layout record ([`HostShape`]), a card
//! count, and the K-alignment the column-sharded weights need. It owns every
//! range a loader or a step shard may ask for, so no other module deals heads
//! or widths by hand. Ranges are dealt in contiguous runs; when a count does
//! not divide, the earlier cards take one extra unit (16 query heads on 3
//! cards deal 6/5/5). A table for one card is the full range everywhere,
//! which is what keeps the single-device path the degenerate case of this
//! arithmetic rather than a second code path.
//!
//! The sharding units, and why:
//! - DeltaNet is dealt by **key-head group**. Each key head owns
//!   `value_heads / key_heads` value heads (2 on Qwen3.6-35B-A3B, a checked
//!   ratio of `HostShape`), and a run never splits a key head's value heads,
//!   so the recurrent state, the conv channels and the `out` columns of a
//!   group stay on one card.
//! - Full attention is dealt by **query head**. The shard unit of `AttnQ` is
//!   one head's fused query-then-gate rows. Qwen3.6 has only 2 KV heads, so
//!   the K/V rows are dealt by value blocks instead of heads and the KV cache
//!   itself stays replicated (the step all-gathers each new K/V row).
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

/// Values per dealt block of an expert's intermediate width and of a K/V row.
const VALUE_BLOCK: usize = 32;

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
    query_heads: Vec<Range<usize>>,
    kv_rows: Vec<Range<usize>>,
    inter: Vec<Range<usize>>,
    shared_inter: Vec<Range<usize>>,
    vocab: Vec<Range<usize>>,
    /// Value heads per key head of the shape the table was built from.
    value_per_key: usize,
    key_dim: usize,
    value_dim: usize,
    head_dim: usize,
    total_key_heads: usize,
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
        let key_heads = deal(d.key_heads, 1, cards, "DeltaNet key heads")?;
        let query_heads = deal(a.heads_q, 1, cards, "attention query heads")?;
        let kv_rows = deal(a.heads_kv * a.head_dim, VALUE_BLOCK, cards, "K/V rows")?;
        let inter = deal(m.inter, VALUE_BLOCK, cards, "expert intermediate")?;
        let shared_inter = deal(m.shared_inter, VALUE_BLOCK, cards, "shared intermediate")?;
        let vocab = deal(shape.vocab, 1, cards, "vocabulary")?;
        let value_per_key = d.value_heads / d.key_heads;
        // Column cuts of the three partial-sum matrices, in K values.
        let delta_out: Vec<Range<usize>> = key_heads
            .iter()
            .map(|run| {
                run.start * value_per_key * d.value_dim..run.end * value_per_key * d.value_dim
            })
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
            query_heads,
            kv_rows,
            inter,
            shared_inter,
            vocab,
            value_per_key,
            key_dim: d.key_dim,
            value_dim: d.value_dim,
            head_dim: a.head_dim,
            total_key_heads: d.key_heads,
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

    /// The key heads `card` runs in every DeltaNet layer.
    pub(crate) fn key_heads(&self, card: usize) -> Result<Range<usize>, EngError> {
        self.at(&self.key_heads, card, "DeltaNet key heads")
    }

    /// The value heads `card` runs: its key-head run widened by the ratio.
    pub(crate) fn value_heads(&self, card: usize) -> Result<Range<usize>, EngError> {
        let run = self.key_heads(card)?;
        Ok(run.start * self.value_per_key..run.end * self.value_per_key)
    }

    /// The three channel segments of the fused DeltaNet `qkv` projection that
    /// belong to `card`, in the stored channel order: its query channels, its
    /// key channels, then its value channels, each as a range inside
    /// `conv_dim` (`[key heads | key heads | value heads]` channels).
    pub(crate) fn delta_qkv_segments(&self, card: usize) -> Result<[Range<usize>; 3], EngError> {
        let keys = self.key_heads(card)?;
        let values = self.value_heads(card)?;
        let key_width = self.total_key_heads * self.key_dim;
        Ok([
            keys.start * self.key_dim..keys.end * self.key_dim,
            key_width + keys.start * self.key_dim..key_width + keys.end * self.key_dim,
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
    fn every_width_partitions_for_one_to_four_cards() {
        let shape = shape();
        for cards in 1..=4 {
            let table = ShardTable::new(&shape, cards, 32).expect("table");
            assert_partition(&runs(&table, ShardTable::key_heads), 16, "key heads");
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
    fn three_cards_deal_the_documented_uneven_runs() {
        let table = ShardTable::new(&shape(), 3, 32).expect("three cards");
        assert_eq!(
            runs(&table, ShardTable::key_heads),
            vec![0..6, 6..11, 11..16]
        );
        assert_eq!(
            runs(&table, ShardTable::value_heads),
            vec![0..12, 12..22, 22..32]
        );
        assert_eq!(
            runs(&table, ShardTable::query_heads),
            vec![0..6, 6..11, 11..16]
        );
        // 16 blocks of 32 dealt 6/5/5: the earlier card takes the extra block.
        assert_eq!(
            runs(&table, ShardTable::expert_inter),
            vec![0..192, 192..352, 352..512]
        );
        // The middle card's qkv segments: key heads 6..11, value heads 12..22.
        assert_eq!(
            table.delta_qkv_segments(1).expect("segments"),
            [768..1408, 2816..3456, 5632..6912]
        );
    }

    #[test]
    fn a_cut_inside_a_quantization_block_is_refused_by_name() {
        // Expert down K is 512 wide: 2 cards cut at 256 and pass at alignment
        // 256; 3 cards would cut at 192 and must refuse.
        ShardTable::new(&shape(), 2, 256).expect("two cards at 256");
        let error = ShardTable::new(&shape(), 3, 256).expect_err("three cards at 256 refuse");
        match error {
            EngError::BadRequest(message) => {
                assert!(message.contains("expert down"), "{message}");
                assert!(message.contains("192"), "{message}");
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
        let table = ShardTable::new(&shape(), 3, 32).expect("table");
        let row: Vec<f32> = (0..248_320)
            .map(|i| ((i * 2_654_435_761usize) % 1000) as f32 * 0.001)
            .collect();
        let mut shards = Vec::new();
        for card in 0..3 {
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
    fn zero_cards_and_a_card_past_the_table_refuse() {
        assert!(ShardTable::new(&shape(), 0, 1).is_err());
        let table = ShardTable::new(&shape(), 2, 1).expect("two cards");
        assert!(table.key_heads(2).is_err());
        assert!(table.head_rows(9).is_err());
    }
}
