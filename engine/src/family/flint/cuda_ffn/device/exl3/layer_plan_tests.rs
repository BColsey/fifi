use super::*;
use crate::family::flint::cuda_ffn::plan::TOP_K;

/// A deterministic router: six distinct experts per row, skewed toward low
/// ids so some experts are hot and some are never chosen.
pub fn routing(rows: usize, seed: u64) -> Vec<i32> {
    let mut state = seed | 1;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut ids = Vec::with_capacity(rows * TOP_K);
    for _ in 0..rows {
        let mut picked: Vec<i32> = Vec::new();
        while picked.len() < TOP_K {
            let id = (next() % 256).min(next() % 256) as i32;
            if !picked.contains(&id) {
                picked.push(id);
            }
        }
        ids.extend(picked);
    }
    ids
}

fn plan_of(ids: &[i32], capacity: usize) -> (Grouping, LayerPlan) {
    let rows = ids.len() / TOP_K;
    let grouping = Grouping::new(ids, rows, 256).unwrap();
    let plan = LayerPlan::new(ids, &grouping, capacity).unwrap();
    (grouping, plan)
}

#[test]
fn tiles_split_evenly_into_the_fewest_tensor_height_runs() {
    for rows in 1..=2000 {
        let split = tiles(100, rows);
        assert_eq!(split.len(), rows.div_ceil(TILE_MAX_ROWS), "rows {rows}");
        assert_eq!(split.iter().map(|tile| tile.rows).sum::<usize>(), rows);
        let (low, high) = split.iter().fold((usize::MAX, 0), |(low, high), tile| {
            (low.min(tile.rows), high.max(tile.rows))
        });
        assert!(high <= TILE_MAX_ROWS && high - low <= 1, "rows {rows}");
        let mut next = 100;
        for tile in &split {
            assert_eq!(tile.first, next, "tiles are contiguous");
            next += tile.rows;
        }
        if rows > TILE_MAX_ROWS {
            // A big expert never leaves a short remainder tile behind: that
            // remainder would drop off the tensor-core kernel.
            assert_eq!(RowClass::of(low).unwrap(), RowClass::Tensor, "rows {rows}");
        }
    }
    assert_eq!(
        tiles(0, 145),
        [
            Tile { first: 0, rows: 73 },
            Tile {
                first: 73,
                rows: 72
            }
        ]
    );
}

#[test]
fn row_classes_follow_the_projection_entry_thresholds() {
    for (rows, class) in [
        (1, RowClass::Small),
        (8, RowClass::Small),
        (9, RowClass::Tile),
        (63, RowClass::Tile),
        (64, RowClass::Tensor),
        (144, RowClass::Tensor),
    ] {
        assert_eq!(RowClass::of(rows).unwrap(), class, "rows {rows}");
    }
    for rows in [0, 145, usize::MAX] {
        assert!(RowClass::of(rows).is_err(), "rows {rows}");
    }
}

#[test]
fn every_pair_runs_exactly_once_expert_major_for_the_tail_heights() {
    for rows in [1, 7, 8, 9, 63, 64, 65, 200, 4000] {
        let ids = routing(rows, 0x5eed + rows as u64);
        let (grouping, plan) = plan_of(&ids, 224);
        assert_eq!(plan.pairs, ids.len());
        let mut seen = vec![false; ids.len()];
        let mut previous_expert = None;
        for visit in plan.rounds.iter().flatten() {
            assert!(previous_expert < Some(visit.id), "experts ascend");
            previous_expert = Some(visit.id);
            let mut previous_pair = None;
            for tile in &visit.tiles {
                assert!(RowClass::of(tile.rows).is_ok());
                for position in tile.first..tile.first + tile.rows {
                    let pair = grouping.order[position];
                    assert_eq!(ids[pair] as usize, visit.id, "rows {rows}");
                    assert!(previous_pair < Some(pair), "rows ascend inside an expert");
                    previous_pair = Some(pair);
                    assert!(
                        !std::mem::replace(&mut seen[pair], true),
                        "pair {pair} twice"
                    );
                }
            }
        }
        assert!(seen.iter().all(|&done| done), "rows {rows}: a pair is lost");
    }
}

#[test]
fn rounds_fill_to_capacity_and_split_only_when_the_slot_arena_is_smaller() {
    let ids = routing(4000, 7);
    let distinct = {
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        sorted.dedup();
        sorted.len()
    };
    assert!(
        distinct > 224,
        "a 4000-row piece touches nearly every expert"
    );
    let (_, plan) = plan_of(&ids, 224);
    assert_eq!(plan.rounds.len(), distinct.div_ceil(224));
    let sizes: Vec<_> = plan.rounds.iter().map(Vec::len).collect();
    assert!(sizes[..sizes.len() - 1].iter().all(|&size| size == 224));
    assert_eq!(sizes.iter().sum::<usize>(), distinct);
    // With every expert resident there is nothing to round.
    let (_, whole) = plan_of(&ids, 256);
    assert_eq!(whole.rounds.len(), 1);
    // A capacity of one makes every expert its own round.
    let (_, single) = plan_of(&routing(9, 3), 1);
    assert!(single.rounds.iter().all(|round| round.len() == 1));
}

#[test]
fn a_hot_expert_is_tiled_and_a_cold_one_is_not() {
    // 400 rows all choosing experts 0..6: each expert owns 400 rows.
    let ids: Vec<i32> = (0..400).flat_map(|_| 0..TOP_K as i32).collect();
    let (_, plan) = plan_of(&ids, 224);
    let visits = &plan.rounds[0];
    assert_eq!(visits.len(), TOP_K);
    for visit in visits {
        assert_eq!(visit.rows(), 400);
        assert_eq!(
            visit.tiles.iter().map(|tile| tile.rows).collect::<Vec<_>>(),
            [134, 133, 133]
        );
    }
}

#[test]
fn planning_refuses_empty_routes_and_a_zero_round_capacity() {
    let ids = routing(9, 1);
    let grouping = Grouping::new(&ids, 9, 256).unwrap();
    assert!(LayerPlan::new(&ids, &grouping, 0).is_err());
    assert!(LayerPlan::new(&ids[..ids.len() - 1], &grouping, 224).is_err());
    assert!(LayerPlan::new(&[], &grouping, 224).is_err());
}

/// A walker that records the rounds and tiles it is asked to run and fails on
/// demand, standing in for the device implementation.
#[derive(Default)]
pub struct Recorder {
    pub rounds: Vec<Vec<i32>>,
    pub tiles: Vec<(usize, usize, Tile)>,
    pub fail_at_tile: Option<usize>,
}

struct Visits<'a>(&'a mut Recorder);

impl ExpertTiles for Visits<'_> {
    fn tile(&mut self, expert: usize, tile: Tile) -> Result<(), EngError> {
        let recorder = &mut *self.0;
        if recorder.fail_at_tile == Some(recorder.tiles.len()) {
            return Err(bad("recorded tile failure"));
        }
        recorder
            .tiles
            .push((recorder.rounds.len() - 1, expert, tile));
        Ok(())
    }
}

impl TileWalk for Recorder {
    fn round(
        &mut self,
        ids: &[i32],
        visit: &mut dyn FnMut(&mut dyn ExpertTiles) -> Result<(), EngError>,
    ) -> Result<(), EngError> {
        self.rounds.push(ids.to_vec());
        visit(&mut Visits(self))
    }
}

#[test]
fn the_walk_stages_each_round_then_visits_its_tiles_in_plan_order() {
    let ids = routing(500, 11);
    let (_, plan) = plan_of(&ids, 100);
    let mut recorder = Recorder::default();
    walk(&mut recorder, &plan).unwrap();
    assert_eq!(recorder.rounds.len(), plan.rounds.len());
    for (index, round) in plan.rounds.iter().enumerate() {
        let mut expected: Vec<i32> = round.iter().map(|visit| visit.id as i32).collect();
        expected.sort_by_key(|&id| plan.last_use(id as usize));
        assert_eq!(recorder.rounds[index], expected);
        let ran: Vec<_> = recorder
            .tiles
            .iter()
            .filter(|(round, _, _)| *round == index)
            .map(|&(_, expert, tile)| (expert, tile))
            .collect();
        let planned: Vec<_> = round
            .iter()
            .flat_map(|visit| visit.tiles.iter().map(|&tile| (visit.id, tile)))
            .collect();
        assert_eq!(ran, planned, "round {index}");
    }
}

#[test]
fn a_failing_tile_stops_the_walk_before_any_later_round_is_staged() {
    let ids = routing(500, 11);
    let (_, plan) = plan_of(&ids, 100);
    assert!(plan.rounds.len() > 2);
    let mut recorder = Recorder {
        fail_at_tile: Some(3),
        ..Recorder::default()
    };
    let error = walk(&mut recorder, &plan).unwrap_err();
    assert_eq!(error, bad("recorded tile failure"));
    assert_eq!(recorder.tiles.len(), 3);
    assert_eq!(recorder.rounds.len(), 1, "no later round was staged");
}

#[test]
fn a_resident_first_plan_visits_every_resident_expert_before_any_that_needs_a_copy() {
    let ids = routing(300, 7);
    // The experts below 200 are on the device; a round holds 200 experts.
    let mut resident = [false; 256];
    resident[..200].iter_mut().for_each(|slot| *slot = true);
    let grouping = Grouping::ranked(&ids, 300, 256, &resident).unwrap();
    let plan = LayerPlan::new(&ids, &grouping, 200).unwrap();
    let visits: Vec<&ExpertVisit> = plan.rounds.iter().flatten().collect();
    let first_copy = visits
        .iter()
        .position(|visit| !resident[visit.id])
        .expect("the routing needs some copies");
    assert!(first_copy > 0, "some resident expert is routed");
    assert!(visits[..first_copy].iter().all(|visit| resident[visit.id]));
    assert!(visits[first_copy..].iter().all(|visit| !resident[visit.id]));
    assert_eq!(plan.rounds.len(), visits.len().div_ceil(200));
    assert_eq!(
        visits.iter().map(|visit| visit.rows()).sum::<usize>(),
        ids.len()
    );
}

#[test]
fn experts_are_staged_and_remembered_by_the_last_row_that_picked_them() {
    // Rows 0..3 pick (5,1,2,3,4,6), (1,7,8,9,10,11), (7,5,12,13,14,15).
    let ids = [5, 1, 2, 3, 4, 6, 1, 7, 8, 9, 10, 11, 7, 5, 12, 13, 14, 15];
    let (_, plan) = plan_of(&ids, 256);
    assert_eq!(plan.last_use(2), Some(0));
    assert_eq!(plan.last_use(1), Some(1));
    assert_eq!(plan.last_use(5), Some(2));
    assert_eq!(plan.last_use(200), None);
    let order = plan.recency();
    assert_eq!(order.len(), 15);
    assert_eq!(&order[..4], &[2, 3, 4, 6]);
    assert_eq!(&order[4..9], &[1, 8, 9, 10, 11]);
    assert_eq!(&order[9..], &[5, 7, 12, 13, 14, 15]);
    // The walk hands a round to the cache in that order.
    let mut recorder = Recorder::default();
    walk(&mut recorder, &plan).unwrap();
    assert_eq!(recorder.rounds.len(), 1);
    let staged: Vec<u16> = recorder.rounds[0].iter().map(|&id| id as u16).collect();
    assert_eq!(staged, order);
}
