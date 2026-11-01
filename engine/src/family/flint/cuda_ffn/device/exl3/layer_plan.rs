//! Host plan of one layer-major routed pass: which experts run in which round
//! and how each expert's rows split into kernel-sized tiles.
//!
//! The plan is pure arithmetic over the router's choices. The device walk
//! below it and the symbolic observer in the tests both follow the same
//! [`walk`], so the order they see cannot drift apart.

use super::bad;
use crate::{
    EngError,
    family::flint::cuda_ffn::plan::{Grouping, TOP_K},
};

/// Rows one tile may carry: the row cap of the tensor-core projection entry.
pub(super) const TILE_MAX_ROWS: usize = 144;

/// Largest row count the split-K small-row kernels accept.
pub(super) const SMALL_MAX_ROWS: usize = 8;
/// First row count the projection entry sends to the tensor-core kernel.
const TENSOR_MIN_ROWS: usize = 64;

/// The kernel family a tile of this many rows reaches. The dispatcher and the
/// plan statistics both read it, so a threshold change shows up in both.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum RowClass {
    Small,
    /// 9..=63 rows: the 16x16 SIMT tile.
    Tile,
    Tensor,
}

impl RowClass {
    pub(super) fn of(rows: usize) -> Result<Self, EngError> {
        if (1..=SMALL_MAX_ROWS).contains(&rows) {
            Ok(Self::Small)
        } else if (SMALL_MAX_ROWS + 1..TENSOR_MIN_ROWS).contains(&rows) {
            Ok(Self::Tile)
        } else if (TENSOR_MIN_ROWS..=TILE_MAX_ROWS).contains(&rows) {
            Ok(Self::Tensor)
        } else {
            Err(bad("expert tile row count is outside 1..=144"))
        }
    }
}

/// A run of one expert's rows inside the expert-sorted pair order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Tile {
    /// Position of the first row in the sorted pair order.
    pub(super) first: usize,
    pub(super) rows: usize,
}

/// Every row routed to one expert in the whole layer.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct ExpertVisit {
    pub(super) id: usize,
    pub(super) tiles: Vec<Tile>,
}

#[cfg(test)]
impl ExpertVisit {
    pub(super) fn rows(&self) -> usize {
        self.tiles.iter().map(|tile| tile.rows).sum()
    }
}

/// Experts that are resident together. A round is as large as the slot arena
/// allows so the layer pays the fewest upload passes.
pub(super) type Round = Vec<ExpertVisit>;

#[derive(Debug, PartialEq, Eq)]
pub(super) struct LayerPlan {
    pub(super) rounds: Vec<Round>,
    pub(super) pairs: usize,
    /// For each expert id, the last row of the piece that picked it.
    last_use: Vec<Option<usize>>,
}

impl LayerPlan {
    /// The last row of the piece that picked `expert`, if any did.
    pub(super) fn last_use(&self, expert: usize) -> Option<usize> {
        self.last_use.get(expert).copied().flatten()
    }

    /// The experts the piece used, ordered by the last row that picked them, earliest first. A
    /// token-by-token walk leaves its cache in this recency order.
    pub(super) fn recency(&self) -> Vec<u16> {
        let mut used: Vec<(usize, u16)> = self
            .last_use
            .iter()
            .enumerate()
            .filter_map(|(expert, row)| row.map(|row| (row, expert as u16)))
            .collect();
        used.sort();
        used.into_iter().map(|(_, expert)| expert).collect()
    }
}

/// Splits `rows` into the fewest tiles of at most [`TILE_MAX_ROWS`], as even
/// as possible: a 145-row expert becomes 73 + 72, never 144 + 1, so every
/// tile of a large expert stays on the tensor-core kernel.
fn tiles(first: usize, rows: usize) -> Vec<Tile> {
    let count = rows.div_ceil(TILE_MAX_ROWS);
    let (base, extra) = (rows / count, rows % count);
    let mut start = first;
    (0..count)
        .map(|index| {
            let rows = base + usize::from(index < extra);
            let tile = Tile { first: start, rows };
            start += rows;
            tile
        })
        .collect()
}

impl LayerPlan {
    /// `grouping` is the stable expert-major order of every (row, rank) pair
    /// of the piece. Within a group of experts (the resident ones first when
    /// the grouping is [`Grouping::ranked`] over the bank's residency) experts
    /// keep ascending id order and rows keep ascending pair order, so the plan
    /// depends only on the router's choices and the residency.
    pub(super) fn new(
        ids: &[i32],
        grouping: &Grouping,
        round_experts: usize,
    ) -> Result<Self, EngError> {
        if round_experts == 0 || grouping.order.len() != ids.len() || ids.is_empty() {
            return Err(bad("layer plan needs routed pairs and a round capacity"));
        }
        let mut visits = Vec::new();
        let mut first = 0;
        while first < ids.len() {
            let id = ids[grouping.order[first]];
            let mut end = first + 1;
            while end < ids.len() && ids[grouping.order[end]] == id {
                end += 1;
            }
            let id = usize::try_from(id).map_err(|_| bad("routed expert id is negative"))?;
            visits.push(ExpertVisit {
                id,
                tiles: tiles(first, end - first),
            });
            first = end;
        }
        let mut rounds = Vec::new();
        let mut visits = visits.into_iter().peekable();
        while visits.peek().is_some() {
            rounds.push(visits.by_ref().take(round_experts).collect());
        }
        let mut last_use = vec![None; ids.iter().copied().max().map_or(0, |id| id as usize + 1)];
        for (pair, &id) in ids.iter().enumerate() {
            if let Some(slot) = usize::try_from(id).ok().and_then(|id| last_use.get_mut(id)) {
                *slot = Some(pair / TOP_K);
            }
        }
        Ok(Self {
            rounds,
            pairs: ids.len(),
            last_use,
        })
    }
}

/// What the walk asks of one resident round: run a tile of one expert.
pub(super) trait ExpertTiles {
    fn tile(&mut self, expert: usize, tile: Tile) -> Result<(), EngError>;

    /// Runs every tile of one resident round. The default visits them one by one; the job-table
    /// pass runs the whole round in a few launches.
    fn round(&mut self, visits: &[ExpertVisit]) -> Result<(), EngError> {
        for visit in visits {
            for tile in &visit.tiles {
                self.tile(visit.id, *tile)?;
            }
        }
        Ok(())
    }
}

/// Makes a round's experts resident for the duration of `visit`. The device
/// implementation stages the weights; the test implementation records.
pub(super) trait TileWalk {
    fn round(
        &mut self,
        ids: &[i32],
        visit: &mut dyn FnMut(&mut dyn ExpertTiles) -> Result<(), EngError>,
    ) -> Result<(), EngError>;
}

/// Runs every tile of the plan, round by round. A tile's rows are contiguous in
/// the sorted pair order, so the device side reads them in place.
pub(super) fn walk(walker: &mut impl TileWalk, plan: &LayerPlan) -> Result<(), EngError> {
    for round in &plan.rounds {
        let mut ids: Vec<i32> = round
            .iter()
            .map(|visit| i32::try_from(visit.id).map_err(|_| bad("expert id overflows")))
            .collect::<Result<_, _>>()?;
        // The cache moves each expert to the front of its recency in the order given, so staging
        // them by the last row that picked them leaves the earliest-used ones to be evicted first.
        ids.sort_by_key(|&id| plan.last_use(id as usize));
        walker.round(&ids, &mut |tiles| tiles.round(round))?;
    }
    Ok(())
}

#[cfg(test)]
#[path = "layer_plan_tests.rs"]
pub(super) mod tests;
