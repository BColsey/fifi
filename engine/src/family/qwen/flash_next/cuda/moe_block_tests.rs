//! Host tests of the split MoE block's logic: the private scratch layout, the classify log, the
//! spans of the bank buffer, and the split of a layer against one pass over an all-resident
//! bank — which the split must equal BIT FOR BIT, at any miss split (the miss pass overwrites
//! the missed ranks' rows and one combine sums every pick in rank order).
//! The expert outputs are synthetic and deterministic; the device test (ignored) runs the
//! real kernels on a real layer.

use super::moe_block::{
    PassInputs, ScratchLayout, SlotSpans, classify_picks, parse_log, pick_union, plan_second_pass,
};
use crate::family::expert_bank::{BankGeometry, ExpertBankState, HostMode, ReplaceOnMissLru};
use crate::{
    EngError,
    family::{
        project::exl3_jobs_plan::EXL3_JOB_WORDS, qwen::flash_next::moe::slot_bank::SlotBankLayout,
    },
};
// The planning items sit in the native-cuda device module; the scratch-size tests above run on
// every host, so only these imports carry the gate.
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
use {
    super::moe_block::{source_offsets, source_plan},
    crate::family::project::cuda::exl3_jobs::PairPlan,
};

const EXPERTS: usize = 512;
const TOP_K: usize = 10;
const HIDDEN: usize = 8;
const RECORD: usize = 1_881_600;

fn layout() -> SlotBankLayout {
    SlotBankLayout::new(2560, 640, [48, 48, 48]).unwrap()
}

/// A one-layer bank holding `held` (slot `s` holds `held[s]`), and the state around it.
fn bank(held: &[u16]) -> ExpertBankState<ReplaceOnMissLru> {
    let geometry = BankGeometry::new(1, EXPERTS, TOP_K, held.len(), RECORD).unwrap();
    let mut row = vec![-1i32; EXPERTS];
    for (slot, &expert) in held.iter().enumerate() {
        row[usize::from(expert)] = slot as i32;
    }
    // The least popular holder is the first victim: count by id, so a low id is evicted last.
    let counts = vec![(0..EXPERTS as u64).rev().collect::<Vec<u64>>()];
    let policy = ReplaceOnMissLru::from_fill(geometry, &[row.as_slice()], &counts, None).unwrap();
    ExpertBankState::with_policy(geometry, HostMode::Exclusive, policy).unwrap()
}

/// The sum the combine kernel forms for one row: each product rounded alone, the picks added in
/// pick order, then the gated shared term, then the residual (`moe_combine.rs`).
fn combine_reference(
    residual: &[f32],
    down: &[Vec<f32>],
    weights: &[f32],
    shared: Option<(f32, &[f32])>,
) -> Vec<f32> {
    (0..residual.len())
        .map(|j| {
            let mut sum = weights[0] * down[0][j];
            for (weight, row) in weights.iter().zip(down).skip(1) {
                sum += weight * row[j];
            }
            if let Some((gate, row)) = shared {
                sum += gate * row[j];
            }
            residual[j] + sum
        })
        .collect()
}

/// Router weights that sum to one, falling by rank.
fn weights() -> Vec<f32> {
    let raw: Vec<f32> = (0..TOP_K).map(|rank| 1.0 / (rank as f32 + 2.0)).collect();
    let total: f32 = raw.iter().sum();
    raw.iter().map(|w| w / total).collect()
}

/// The down output of `expert` at column `j`, as a deterministic finite f32.
fn down_of(expert: u16, j: usize) -> f32 {
    ((usize::from(expert) * 31 + j * 7) % 97) as f32 / 13.0 - 3.5
}

fn down_rows(experts: impl Iterator<Item = u16>) -> Vec<Vec<f32>> {
    experts
        .map(|expert| (0..HIDDEN).map(|j| down_of(expert, j)).collect())
        .collect()
}

fn shared_row() -> Vec<f32> {
    (0..HIDDEN).map(|j| 0.25 + j as f32 * 0.125).collect()
}

fn holders(state: &ExpertBankState<ReplaceOnMissLru>, slots: usize) -> Vec<u16> {
    (0..slots)
        .map(|slot| state.expert_in_slot(0, slot).expect("a filled slot"))
        .collect()
}

/// Runs the split of one layer on the host: the picks go through the classifier and the policy,
/// then the device order is mirrored — the resident rows hold the pre-copy experts, the missed
/// ranks' rows are overwritten with the post-copy experts, and one combine sums every rank with
/// the router's weights.
/// Returns the split result and the pass inputs.
fn split(
    state: &mut ExpertBankState<ReplaceOnMissLru>,
    ids: &[i32],
    weights: &[f32],
    gate: f32,
) -> (Vec<f32>, PassInputs) {
    let slots = state.geometry().slots;
    let before = holders(state, slots);
    let row_before = state.slot_row(0).to_vec();
    let classified = classify_picks(&row_before, slots, ids, weights).unwrap();
    state.begin_pass(0, ids).unwrap();
    let after = holders(state, slots);
    let second = plan_second_pass(ids, &classified, state.slot_row(0)).unwrap();
    // The device order, mirrored: the resident pass writes every rank row (a filler row at a
    // missed rank), the miss pass overwrites the missed ranks' rows, and ONE combine sums every
    // rank in pick order with the router's true weights.
    let mut experts: Vec<u16> = classified
        .slot_ids
        .iter()
        .map(|&s| before[s as usize])
        .collect();
    if let Some((miss_slots, miss_ranks)) = &second.miss {
        for (j, &rank) in miss_ranks.iter().enumerate() {
            experts[rank as usize] = after[miss_slots[j] as usize];
        }
    }
    let down = down_rows(experts.iter().copied());
    let result = combine_reference(&[0.0; HIDDEN], &down, weights, Some((gate, &shared_row())));
    (result, second)
}

/// The same picks over a bank that holds every expert: one combine in pick order.
fn all_resident(ids: &[i32], weights: &[f32], gate: f32) -> Vec<f32> {
    let down = down_rows(ids.iter().map(|&id| id as u16));
    combine_reference(&[0.0; HIDDEN], &down, weights, Some((gate, &shared_row())))
}

#[test]
fn the_private_scratch_is_aligned_disjoint_and_small() {
    let layout = ScratchLayout::new(512, 10, 2560, 1).unwrap();
    let regions = [
        (layout.slot_row, 512 * 4),
        (layout.slot_ids, 40),
        (layout.weights, 40),
        (layout.log, 80),
        (layout.ctrl, 64),
        (layout.word, 4),
        (layout.mbox_req, 128),
        (layout.mbox_row, 2560 * 4),
        (layout.miss_ids, 40),
        (layout.miss_weights, 40),
        (layout.partial, 2560 * 4),
        // The job-table perm region: the two sources' lists and job tables together —
        // 2 list words plus one 4-word job record per pair (a source's jobs are at most its
        // pairs) — at routed = rows·top_k = 10.
        (layout.perm, 240),
    ];
    for (index, &(offset, bytes)) in regions.iter().enumerate() {
        assert_eq!(offset % 256, 0, "region {index}");
        assert!(offset + bytes <= layout.total, "region {index}");
        for &(other, other_bytes) in &regions[index + 1..] {
            assert!(offset + bytes <= other || other + other_bytes <= offset);
        }
    }
    assert_eq!(layout.total, 24_832, "24.25 KiB at the real sizes");
}

/// The perm region is the last region and covers the widest split of the chunk's pairs over
/// the two sources even when every pair is its own job; the allocation grows with the planned
/// chunk width (the row-indexed regions scale with it).
#[test]
fn the_private_scratch_covers_the_widest_chunk_plan() {
    let one = ScratchLayout::new(512, 10, 2560, 1).unwrap();
    for (rows, routed) in [(32usize, 320usize), (64, 640)] {
        let layout = ScratchLayout::new(512, 10, 2560, rows).unwrap();
        assert_eq!(layout.rows, rows);
        // The perm region is last; its size is 2 list words plus one job record (4 words) per
        // pair — the worst case is every pair in one source with a job of its own, and the two
        // sources sum to the same bound.
        assert_eq!(
            layout.total - layout.perm,
            (2 + EXL3_JOB_WORDS) * routed * 4
        );
        assert!(layout.total > one.total);
        // The row-indexed regions grow by more than the perm region alone.
        assert!(layout.total - one.total > (routed - 10) * (2 + EXL3_JOB_WORDS) * 4);
    }
}

/// One source's counting sort: sorted positions map back to the chunk's pair positions, and
/// a slot id outside the source's slots refuses the chunk.
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
#[test]
fn source_plan_maps_sorted_positions_to_chunk_pairs() {
    // A chunk of 2 rows, top_k 2: pair = row·2 + rank. Pairs 0, 2 and 3 pick bank slots
    // (5, 5, 0); pair 1 picks the ring (slot 9).
    let ids = [5i32, 9, 5, 0];
    let bank = source_plan(&ids, &[0, 2, 3], 8, 2).unwrap();
    assert_eq!(bank.counts, {
        let mut counts = vec![0usize; 8];
        counts[0] = 1;
        counts[5] = 2;
        counts
    });
    // Slot 0 owns pair 3 (row 1); slot 5 owns pairs 0 and 2 (rows 0 and 1), chunk order kept.
    assert_eq!(bank.order, [1, 0, 1]);
    assert_eq!(bank.scatter, [3, 0, 2]);
    let ring = source_plan(&ids, &[1], 16, 2).unwrap();
    assert_eq!(ring.counts, {
        let mut counts = vec![0usize; 16];
        counts[9] = 1;
        counts
    });
    assert_eq!(ring.order, [0]);
    assert_eq!(ring.scatter, [1]);
    // A slot id outside the source's slots is a router fault, refused.
    assert!(matches!(
        source_plan(&ids, &[1], 8, 2),
        Err(EngError::ModelOutput(_))
    ));
}

/// The two sources' lists and job tables pack into the perm region back to back, and their
/// gate/up segments tile the pair faces in source order.
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
#[test]
fn source_offsets_pack_the_perm_region() {
    // Counts drive the job tables: one job per expert with rows (the 64-row tile never splits
    // these small counts), so the bank (3 + 1 rows over two experts) takes two jobs.
    let bank = PairPlan {
        counts: vec![3, 1],
        order: vec![],
        scatter: vec![],
    };
    let ring = PairPlan {
        counts: vec![2],
        order: vec![],
        scatter: vec![],
    };
    let (sub_bank, sub_ring) = ([0usize, 1, 2, 3], [4usize, 5]);
    let (at_of, seg_of) = source_offsets(1_000, [&bank, &ring], [&sub_bank, &sub_ring]);
    // Bank: 4 pairs and 2 jobs — (2·4 + 4·2)·4 bytes of lists and table.
    assert_eq!(at_of, [1_000, 1_000 + (2 * 4 + EXL3_JOB_WORDS * 2) * 4]);
    assert_eq!(seg_of, [0, 4]);
    // Ring: 2 pairs and 1 job; the packing ends inside the perm formula's bound
    // (2 words + one job record a pair over all 6 pairs, 3 jobs).
    let end = at_of[1] + (2 * 2 + EXL3_JOB_WORDS) * 4;
    assert_eq!(end - 1_000, (2 * 6 + EXL3_JOB_WORDS * 3) * 4);
    assert!(end - 1_000 <= (2 + EXL3_JOB_WORDS) * 6 * 4);
}

#[test]
fn the_classify_log_round_trips() {
    let ids: Vec<i32> = vec![5, 400, 7, 2, 511, 0, 31, 9, 8, 100];
    let weights = weights();
    let mut raw = Vec::new();
    for (id, weight) in ids.iter().zip(&weights) {
        raw.extend_from_slice(&id.to_le_bytes());
        raw.extend_from_slice(&weight.to_bits().to_le_bytes());
    }
    let (got_ids, got_weights) = parse_log(&raw, TOP_K).unwrap();
    assert_eq!(got_ids, ids);
    assert!(
        got_weights
            .iter()
            .zip(&weights)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
    assert!(parse_log(&raw[..raw.len() - 1], TOP_K).is_err());
    assert!(parse_log(&raw, TOP_K - 1).is_err());
}

#[test]
fn a_router_fault_and_an_invalid_route_have_no_second_pass() {
    let held: Vec<u16> = (0..16).collect();
    let state = bank(&held);
    let row = state.slot_row(0).to_vec();
    let good: Vec<i32> = (0..10).collect();
    assert!(classify_picks(&row, 16, &good, &weights()).is_ok());
    let mut fault = good.clone();
    fault[3] = -1;
    let error = classify_picks(&row, 16, &fault, &weights()).unwrap_err();
    assert!(matches!(error, EngError::BadRequest(_)), "{error:?}");
    let mut twice = good.clone();
    twice[4] = 2;
    assert!(classify_picks(&row, 16, &twice, &weights()).is_err());
    let mut outside = good.clone();
    outside[0] = 512;
    assert!(classify_picks(&row, 16, &outside, &weights()).is_err());
    assert!(classify_picks(&row, 16, &good, &weights()[..9]).is_err());
}

#[test]
fn the_spans_address_the_bank_buffer_at_the_record_stride() {
    let layout = layout();
    assert_eq!(layout.record_bytes(), RECORD);
    for slots in [11usize, 16, 32] {
        let spans = SlotSpans::new(&layout, slots).unwrap();
        assert_eq!(spans.trellis_stride, RECORD);
        assert_eq!(spans.scale_stride, RECORD / 4);
        let bank_bytes = layout.bank_bytes(slots).unwrap();
        assert_eq!(bank_bytes, slots * RECORD);
        for (index, part) in spans.parts.iter().enumerate() {
            assert_eq!(
                part.shape,
                if index == 2 { [640, 2560] } else { [2560, 640] }
            );
            assert_eq!(part.bits_twice, 6, "3 bits per weight");
            for range in [&part.trellis, &part.suh, &part.svh] {
                assert!(range.end <= bank_bytes, "slots {slots} part {index}");
            }
            // Member `i` of the span is slot `i`'s component: the last member ends the span.
            let own = layout.part(index);
            assert_eq!(
                part.trellis.end - part.trellis.start,
                own.trellis.len() + (slots - 1) * RECORD
            );
            assert_eq!(part.suh.start, own.suh.start);
            assert_eq!(part.svh.start, own.svh.start);
        }
        // The three parts do not overlap within one record.
        let one = &spans.parts;
        assert!(one[0].svh.start < one[1].trellis.start);
        assert!(one[1].svh.start < one[2].trellis.start);
    }
    assert!(SlotSpans::new(&layout, 0).is_err());
}

#[test]
fn all_resident_picks_are_one_pass() {
    let held: Vec<u16> = (0..16).collect();
    let mut state = bank(&held);
    let ids: Vec<i32> = vec![3, 9, 0, 14, 1, 7, 12, 5, 2, 15];
    let w = weights();
    let (result, inputs) = split(&mut state, &ids, &w, 0.6);
    assert_eq!(inputs.misses, 0);
    assert!(inputs.miss.is_none());
    // A layer without a miss is one launch over the same slots: the resident pass IS the result,
    // and with the identity fill it is bit for bit the all-resident combine.
    assert_eq!(result, all_resident(&ids, &w, 0.6));
}

#[test]
fn a_miss_pass_adds_each_missed_pick_once() {
    let held: Vec<u16> = (0..16).collect();
    let mut state = bank(&held);
    // Eight resident picks and two cold ones, in rank positions 2 and 7.
    let ids: Vec<i32> = vec![3, 9, 300, 14, 1, 7, 12, 411, 2, 15];
    let w = weights();
    let classified = classify_picks(state.slot_row(0), 16, &ids, &w).unwrap();
    assert_eq!(classified.misses.len(), 2);
    let (result, inputs) = split(&mut state, &ids, &w, 0.6);
    let (slots, ranks) = inputs.miss.expect("two misses");
    // The compact list carries the missed picks (rank positions 2 and 7) in pick order, and
    // names the rank rows their down outputs overwrite; every other pick keeps its resident
    // row, so the one combine sees each pick exactly once at its true weight.
    assert_eq!(slots.len(), 2);
    assert_eq!(ranks, vec![2, 7]);
    for (pick, &weight) in w.iter().enumerate() {
        let missed = pick == 2 || pick == 7;
        assert_eq!(classified.weights[pick] != 0.0, !missed, "pick {pick}");
        // The combine weight is the router's weight bit for bit at every rank.
        assert_eq!(
            classified.log[pick].1.to_bits(),
            weight.to_bits(),
            "pick {pick}"
        );
    }
    // The missed picks sit on slots the copies filled, never on slot 0.
    for (entry, &pick) in [2usize, 7].iter().enumerate() {
        assert_ne!(slots[entry], 0);
        assert_eq!(
            state.expert_in_slot(0, slots[entry] as usize),
            Some(ids[pick] as u16)
        );
    }
    // The split IS one pass over all-resident experts, bit for bit: one rank-order combine.
    assert_eq!(result, all_resident(&ids, &w, 0.6));
}

#[test]
fn every_pick_missing_is_all_filler_then_all_picks() {
    // The bank holds sixteen experts none of which the router picks.
    let held: Vec<u16> = (100..116).collect();
    let mut state = bank(&held);
    let ids: Vec<i32> = vec![3, 9, 300, 14, 1, 7, 12, 411, 2, 15];
    let w = weights();
    let classified = classify_picks(state.slot_row(0), 16, &ids, &w).unwrap();
    assert_eq!(classified.misses.len(), 10);
    assert!(classified.slot_ids.iter().all(|&s| s == 0));
    assert!(classified.weights.iter().all(|w| w.to_bits() == 0));
    let (result, inputs) = split(&mut state, &ids, &w, 0.6);
    let (slots, ranks) = inputs.miss.expect("ten misses");
    assert_eq!(inputs.misses, 10);
    // Ten picks of ten distinct experts need ten slots besides slot 0.
    let mut seen = slots.clone();
    seen.sort_unstable();
    seen.dedup();
    assert_eq!(seen.len(), 10);
    assert!(!slots.contains(&0));
    // Every rank row is overwritten: the all-filler resident pass contributes nothing that
    // survives to the one combine.
    assert_eq!(ranks, (0..10u32).collect::<Vec<u32>>());
    assert_eq!(result, all_resident(&ids, &w, 0.6));
}

#[test]
fn the_filler_experts_identity_cannot_reach_the_sum() {
    // Replaces the removed filler-NaN tests. Slot 0 is the designated filler: when a pick
    // misses, the resident pass reads slot 0's rows at the missed ranks while the copies run
    // (plan_pass protects slot 0 for exactly that), and the miss pass overwrites those rows
    // before the one combine. Two banks differing ONLY in slot 0's expert must therefore
    // write identical bytes.
    let ids: Vec<i32> = vec![3, 9, 300, 14, 1, 7, 12, 411, 2, 15];
    let w = weights();
    let resident: Vec<u16> = [3u16, 9, 14, 1, 7, 12, 2, 15].to_vec();
    let tail: Vec<u16> = [101u16, 102, 103, 104, 105, 106, 107].to_vec();
    let held_a: Vec<u16> = [vec![500u16], resident.clone(), tail.clone()].concat();
    let held_b: Vec<u16> = [vec![501u16], resident, tail].concat();
    let mut state_a = bank(&held_a);
    let mut state_b = bank(&held_b);
    let (result_a, inputs_a) = split(&mut state_a, &ids, &w, 0.6);
    let (result_b, inputs_b) = split(&mut state_b, &ids, &w, 0.6);
    assert_eq!(inputs_a.misses, 2);
    assert_eq!(inputs_b.misses, 2, "the same picks miss in both banks");
    assert!(
        !inputs_a.miss.as_ref().unwrap().0.contains(&0)
            && !inputs_b.miss.as_ref().unwrap().0.contains(&0),
        "slot 0 is the protected filler, never a victim"
    );
    assert_eq!(
        result_a, result_b,
        "the filler expert cannot change the bytes"
    );
    assert_eq!(result_a, all_resident(&ids, &w, 0.6));
}

#[test]
fn a_second_use_of_the_same_picks_is_one_pass() {
    let held: Vec<u16> = (0..16).collect();
    let mut state = bank(&held);
    let ids: Vec<i32> = vec![3, 9, 300, 14, 1, 7, 12, 411, 2, 15];
    let w = weights();
    let (first, inputs) = split(&mut state, &ids, &w, 0.6);
    assert_eq!(inputs.misses, 2);
    let (again, inputs) = split(&mut state, &ids, &w, 0.6);
    assert_eq!(inputs.misses, 0, "the missed experts now sit in the bank");
    assert_eq!(
        again, first,
        "a clean pass and a split pass sum identically"
    );
}

#[test]
fn the_bank_state_after_a_pass_is_what_the_second_pass_reads() {
    let held: Vec<u16> = (0..11).collect();
    let mut state = bank(&held);
    let ids: Vec<i32> = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 200];
    let w = weights();
    let row_before = state.slot_row(0).to_vec();
    let classified = classify_picks(&row_before, 11, &ids, &w).unwrap();
    let copies = state.begin_pass(0, &ids).unwrap();
    assert_eq!((copies.hits, copies.ops.len()), (9, 1));
    let second = plan_second_pass(&ids, &classified, state.slot_row(0)).unwrap();
    let (slots, ranks) = second.miss.unwrap();
    // Pick 9 is the only miss, so the compact list holds it at index 0 and overwrites its rank.
    assert_eq!(slots[0], i32::from(copies.ops[0].slot));
    assert_eq!(ranks, vec![9]);
    // The classification made before the copies is stale after them: a second use of the old row
    // would send the missed pick to the filler.
    assert_eq!(row_before[200], -1);
    assert!(state.slot_row(0)[200] >= 0);
}

#[test]
fn pick_union_keeps_each_distinct_id_first_use_first() {
    let rows: Vec<&[i32]> = vec![&[7, 3, 7], &[3, 9], &[]];
    assert_eq!(pick_union(&rows, 16), vec![7, 3, 9]);
    // Duplicates inside one row and across rows collapse to the first use.
    assert_eq!(pick_union(&[&[5, 0, 5], &[0, 5]], 8), vec![5, 0]);
}

#[test]
fn slot_spans_fit_a_ring_smaller_than_the_bank_slot_row() {
    let layout = layout();
    let record = layout.record_bytes();
    // A ring's spans over the SAME layout with FEWER slots, checked against the layout's own
    // arithmetic - not against `bank_ranges`, which is what `SlotSpans::new` calls: each
    // part's band starts at its record-0 offset and ends at that offset plus the part's own
    // bytes plus one record stride per extra slot, so the whole band fits the ring's
    // `slots * record_bytes` buffer, and a one-slot ring keeps exactly the base record ranges.
    for slots in [1usize, 2, 7, 16] {
        let spans = SlotSpans::new(&layout, slots).unwrap();
        for part in 0..3 {
            let base = layout.part(part);
            let bands = [&base.trellis, &base.suh, &base.svh];
            let got = [
                &spans.parts[part].trellis,
                &spans.parts[part].suh,
                &spans.parts[part].svh,
            ];
            for (band, got) in bands.iter().zip(got) {
                let (band, got) = (
                    band as &std::ops::Range<usize>,
                    got as &std::ops::Range<usize>,
                );
                assert_eq!(
                    got.start, band.start,
                    "part {part} band start at {slots} slots"
                );
                assert_eq!(
                    got.end,
                    band.end + (slots - 1) * record,
                    "part {part} band end at {slots} slots"
                );
                assert!(
                    got.end <= record * slots,
                    "part {part} band at {slots} slots leaves the ring buffer"
                );
                if slots == 1 {
                    assert_eq!(got, band, "part {part} one-slot ring keeps the base range");
                }
            }
        }
    }
    // Zero slots is a refusal, the same one the layout's bank_ranges gives.
    assert!(SlotSpans::new(&layout, 0).is_err());
}
