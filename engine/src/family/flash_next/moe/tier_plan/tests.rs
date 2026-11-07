//! The tier plan at the sizes of the 3.05 bpw pack on a 24 GiB card, against the cost model's
//! slot counts (the residency cost model: 143 slots at 262,144 tokens, 178 to 184
//! at 131,072, 206 to 212 at 32,768, about 174 with int8 K/V at 262,144), and the hot set against
//! the classifier oracle at ten picks.

use super::*;

const GIB: u64 = 1 << 30;
const MIB: u64 = 1 << 20;
/// Routed experts of the 3.05 pack: 24,576 of 1,881,600 device bytes each.
const EXPERT_BYTES: u64 = 1_881_600;
/// Header-walk trunk: attention, DeltaNet, router, shared expert, hyper-connections and head.
const TRUNK_BYTES: u64 = 3_854_000_000;
/// 36 DeltaNet layers of 3 MiB of state.
const STATE_BYTES: u64 = 36 * 3 * MIB;

fn shape() -> TierShape {
    TierShape::from_pack(48, 512, 10, 24_576 * EXPERT_BYTES).unwrap()
}

/// K/V and indexer bytes of one position: 12 QSA layers, 2 KV heads of 256, indexer 128 per 4 positions.
fn position(elem: u64) -> u64 {
    position_bytes(12, 2, 256, elem, (128, 4)).unwrap()
}

fn inputs(positions: u64, elem: u64) -> TierInputs {
    TierInputs::rtx3090(
        shape(),
        (TRUNK_BYTES, STATE_BYTES),
        KvPlan {
            positions,
            bytes_per_position: position(elem),
        },
    )
}

fn slots(positions: u64, elem: u64) -> usize {
    TierPlan::new(&inputs(positions, elem))
        .unwrap()
        .slots_per_layer()
}

fn message(error: EngError) -> String {
    error.describe()
}

#[test]
fn the_shape_follows_the_pack_totals() {
    assert_eq!(shape().expert_bytes, 1_881_600);
    assert_eq!(shape().slot_cost(), 48 * 1_881_600);
    assert!(TierShape::from_pack(48, 512, 10, 24_576 * EXPERT_BYTES + 1).is_err());
    assert!(TierShape::from_pack(0, 512, 10, 0).is_err());
}

#[test]
fn a_position_costs_what_the_memory_budget_says() {
    // 24 KiB of f16 K/V plus 1.5 KiB of indexer keys per position; 6.375 GiB at 262,144.
    assert_eq!(position(2), 26_112);
    assert_eq!(262_144 * position(2), 6 * GIB + 3 * GIB / 8);
    assert_eq!(position(1), 13_824);
    assert!(position_bytes(12, 2, 256, 2, (128, 3)).is_err());
    assert!(position_bytes(12, 2, 256, 2, (128, 0)).is_err());
}

#[test]
fn the_slot_counts_track_the_cost_model_at_each_context() {
    let near = |got: usize, want: usize, by: usize| got.abs_diff(want) <= by;
    let (long, mid, short, int8) = (
        slots(262_144, 2),
        slots(131_072, 2),
        slots(32_768, 2),
        slots(262_144, 1),
    );
    assert!(near(long, 143, 3), "262,144 tokens f16: {long}");
    assert!((178..=184).contains(&mid), "131,072 tokens f16: {mid}");
    assert!((206..=212).contains(&short), "32,768 tokens f16: {short}");
    assert!(near(int8, 174, 4), "262,144 tokens int8: {int8}");
    assert!(
        long < mid && mid < short,
        "slots shrink as the context grows"
    );
}

#[test]
fn the_plan_accounts_for_every_byte() {
    let plan = TierPlan::new(&inputs(131_072, 2)).unwrap();
    let slots = plan.slots_per_layer() as u64;
    assert_eq!(plan.bank_bytes(), slots * 48 * EXPERT_BYTES);
    assert_eq!(plan.cold_bytes(), (512 - slots) * 48 * EXPERT_BYTES);
    let i = inputs(131_072, 2);
    let committed = i.vram.non_expert + i.vram.state + i.vram.arena + i.vram.desktop_reserve;
    assert_eq!(
        committed + 131_072 * position(2) + plan.bank_bytes() + plan.vram_left(),
        24 * GIB
    );
    assert!(plan.vram_left() < plan.shape().slot_cost());
    assert_eq!(
        plan.host_spare_bytes(),
        (62 - 8 - 2 - 1 - 1) * GIB - plan.cold_bytes()
    );
}

#[test]
fn a_small_model_is_fully_resident() {
    let mut small = inputs(1_024, 2);
    small.shape = TierShape::from_pack(4, 16, 10, 64 * EXPERT_BYTES).unwrap();
    let plan = TierPlan::new(&small).unwrap();
    assert_eq!(plan.slots_per_layer(), 16, "capped at the expert count");
    assert_eq!(plan.cold_bytes(), 0);
}

#[test]
fn a_context_that_does_not_fit_is_refused_by_name() {
    // The K/V never spills to the host: 2 million positions are 48 GiB of it.
    let text = message(TierPlan::new(&inputs(2_000_000, 2)).unwrap_err());
    assert!(text.contains("KvDoesNotFit"), "{text}");
    // A context that leaves room for fewer than top_k (10) slots in every layer.
    let positions =
        (24 * GIB - TRUNK_BYTES - STATE_BYTES - 1976 * MIB - 9 * 48 * EXPERT_BYTES) / position(2);
    let text = message(TierPlan::new(&inputs(positions, 2)).unwrap_err());
    assert!(
        text.contains("KvDoesNotFit") || text.contains("minimum expert bank"),
        "{text}"
    );
    // A trunk that leaves nothing.
    let mut full = inputs(1_024, 2);
    full.vram.non_expert = 24 * GIB;
    let text = message(TierPlan::new(&full).unwrap_err());
    assert!(text.contains("Overcommitted"), "{text}");
}

#[test]
fn cold_experts_that_exceed_the_pinned_host_budget_are_refused() {
    let mut tight = inputs(262_144, 2);
    tight.host.total = 40 * GIB;
    let text = message(TierPlan::new(&tight).unwrap_err());
    assert!(text.contains("pinned host bytes"), "{text}");
    let mut gone = inputs(262_144, 2);
    gone.host.total = 8 * GIB;
    let text = message(TierPlan::new(&gone).unwrap_err());
    assert!(text.contains("Overcommitted"), "{text}");
    let mut wide = inputs(1_024, 2);
    wide.shape.top_k = 513;
    assert!(TierPlan::new(&wide).is_err());
}

/// Counts that favour experts by a fixed rule so the hot set is known: expert `e` of layer `l` scores
/// `(e * 7 + l * 13) % 512`, a permutation of 0..512 in every layer, so every score is distinct.
fn counts() -> Vec<Vec<u64>> {
    (0..48)
        .map(|layer| {
            (0..512)
                .map(|e| ((e * 7 + layer * 13) % 512) as u64)
                .collect()
        })
        .collect()
}

#[test]
fn the_hot_set_takes_the_hottest_experts_of_every_layer() {
    let plan = TierPlan::new(&inputs(131_072, 2)).unwrap();
    let slots = plan.slots_per_layer();
    let scores = counts();
    let hot = plan.hot_set(&scores).unwrap();
    assert_eq!(hot.table.slots(), slots);
    for (layer, layer_scores) in scores.iter().enumerate() {
        let row = hot.table.row(layer);
        assert_eq!(row.len(), 512);
        let held: Vec<usize> = (0..512).filter(|&e| row[e] >= 0).collect();
        assert_eq!(held.len(), slots, "layer {layer}");
        // Slots are 0..slots in expert-id order, each used once.
        let slot_ids: Vec<i32> = held.iter().map(|&e| row[e]).collect();
        assert_eq!(slot_ids, (0..slots as i32).collect::<Vec<_>>());
        // Every holder outscores every non-holder.
        let score = |e: usize| layer_scores[e];
        let weakest = held.iter().map(|&e| score(e)).min().unwrap();
        let strongest_cold = (0..512).filter(|e| row[*e] < 0).map(score).max();
        assert!(
            strongest_cold.is_none_or(|cold| cold < weakest),
            "layer {layer}"
        );
        for (expert, &slot) in row.iter().enumerate() {
            let want = if slot >= 0 {
                Placement::Gpu
            } else {
                Placement::Cpu
            };
            assert_eq!(
                hot.residency.placement(ExpertId {
                    layer: layer as u32,
                    expert: expert as u32
                }),
                want
            );
        }
    }
    assert_eq!(hot.residency.resident_experts(), slots * 48);
    assert!(hot.residency.full_layers().is_empty());
}

#[test]
fn ties_go_to_the_lowest_expert_id_and_the_profile_must_match_the_shape() {
    let plan = TierPlan::new(&inputs(131_072, 2)).unwrap();
    let flat = vec![vec![5u64; 512]; 48];
    let hot = plan.hot_set(&flat).unwrap();
    let held: Vec<usize> = (0..512).filter(|&e| hot.table.row(0)[e] >= 0).collect();
    assert_eq!(held, (0..plan.slots_per_layer()).collect::<Vec<_>>());
    assert!(plan.hot_set(&flat[..47]).is_err(), "a layer is missing");
    let mut short = flat.clone();
    short[3].pop();
    assert!(plan.hot_set(&short).is_err(), "a layer is short");
}

#[test]
fn ten_picks_classify_against_the_slot_table() {
    let plan = TierPlan::new(&inputs(131_072, 2)).unwrap();
    let hot = plan.hot_set(&counts()).unwrap();
    let layer = 5;
    let row = hot.table.row(layer);
    let resident: Vec<usize> = (0..512).filter(|&e| row[e] >= 0).collect();
    let cold: Vec<usize> = (0..512).filter(|&e| row[e] < 0).collect();
    assert!(resident.len() >= 6 && cold.len() >= 4);
    // Ten picks: six resident (in an interleaved order) and four cold.
    let picks = [
        resident[0],
        cold[0],
        resident[7],
        resident[1],
        cold[1],
        resident[2],
        cold[2],
        resident[3],
        cold[3],
        resident[9],
    ];
    let ids: Vec<i32> = picks.iter().map(|&e| e as i32).collect();
    let weights: Vec<f32> = (1..=10).map(|k| k as f32 / 55.0).collect();
    let got = hot.table.classify(layer, &ids, &weights).unwrap();
    assert_eq!(got.slot_ids.len(), 10);
    assert_eq!(got.word, 0);
    let missed: Vec<i32> = got.misses.iter().map(|&(id, _)| id).collect();
    assert_eq!(
        missed,
        vec![
            cold[0] as i32,
            cold[1] as i32,
            cold[2] as i32,
            cold[3] as i32
        ]
    );
    for (index, &expert) in picks.iter().enumerate() {
        match hot.table.slot_of(layer, expert) {
            Some(slot) => {
                assert_eq!(got.slot_ids[index], slot as i32);
                assert_eq!(got.weights[index].to_bits(), weights[index].to_bits());
            }
            None => {
                assert_eq!(got.slot_ids[index], 0, "a miss reads slot 0");
                assert_eq!(got.weights[index].to_bits(), 0.0f32.to_bits());
            }
        }
    }
    assert_eq!(hot.table.slot_of(layer, 512), None);
    assert!(hot.table.classify(layer, &ids, &weights[..9]).is_err());
}

#[test]
fn route_counts_record_picks_and_refuse_ids_outside_the_layer() {
    let mut counts = RouteCounts::new(2, 8);
    counts.record(1, &[3, 3, 7]).unwrap();
    counts.record(0, &[0]).unwrap();
    assert_eq!(counts.counts()[1][3], 2);
    assert_eq!(counts.counts()[1][7], 1);
    assert_eq!(counts.counts()[0][0], 1);
    assert!(counts.record(2, &[0]).is_err());
    assert!(counts.record(0, &[8]).is_err());
}
