//! The cache cells every session of one executor shares.
//!
//! An executor that allocates a cache per session used to cut the card into equal sessions at
//! start: a prompt above one session's cells was refused while the rest of the card sat idle. A
//! pool holds the cells the card carries once. Each session leases the cells its request needs,
//! up to a per-session ceiling; a request the pool cannot fund yet queues until a session lets
//! go. Pure arithmetic: the executor owns the device planes and asks the pool only how many
//! cells it may allocate.

#![cfg_attr(
    not(feature = "native-cuda"),
    allow(dead_code, reason = "the CUDA-build executors lease from the pool")
)]

use crate::EngError;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

/// The step a session's cells are rounded up to.
pub(crate) const CELL_GRANULE: usize = 1024;

/// The positions a request without a completion limit may generate after its prompt, at least.
pub(crate) const OPEN_COMPLETION_CELLS: usize = 4096;

/// The cells a session holds before a request sizes it.
pub(crate) const INITIAL_CELLS: usize = CELL_GRANULE;

/// The room a request gets beyond its prompt when it is admitted, whatever generation limit it
/// names: a session holds the cells of the tokens it has and grows as it decodes, so a request
/// that may generate 32K tokens does not reserve them up front.
pub(crate) fn admission_room(completion_limit: Option<u32>) -> u32 {
    let room = INITIAL_CELLS as u32;
    completion_limit.map_or(room, |limit| limit.min(room))
}

/// The cells a session that holds `current` grows to when it needs room for `needed` positions:
/// half again what it holds, so a long decode grows a few times and not at every
/// token, rounded to whole granules, never more than `ceiling` and never fewer than it needs.
pub(crate) fn grow_target(current: usize, needed: usize, ceiling: usize) -> usize {
    let wanted = needed.max(current.saturating_add(current / 2));
    wanted
        .div_ceil(CELL_GRANULE)
        .saturating_mul(CELL_GRANULE)
        .min(ceiling)
        .max(needed.min(ceiling))
}

/// Funds the growth of session `who`, which holds `held` cells, to room for `positions`: leases
/// the cells of [`grow_target`] from the pool, or, when the pool cannot fund that much, only what
/// the positions need. Returns the cells the lease now holds. A pool that cannot fund even that
/// refuses with a memory-plan error naming the session and the shortfall and the lease is as it
/// was; so is a request for more than one session can hold. The one rule of every family that
/// grows its sessions on the pool.
pub(crate) fn fund_growth(
    lease: &mut CellLease,
    who: impl std::fmt::Display,
    held: usize,
    positions: usize,
    ceiling: usize,
) -> Result<usize, EngError> {
    if positions > ceiling {
        return Err(EngError::ContextLimit(format!(
            "session {who}: {positions} positions exceed the {ceiling} cells of one session"
        )));
    }
    let target = grow_target(held, positions, ceiling);
    if lease.resize_for(target, ceiling).is_ok() {
        return Ok(target);
    }
    let least = grow_target(held, positions, positions);
    lease.resize_for(least, ceiling).map_err(|_| {
        EngError::MemoryPlan(format!(
            "session {who} needs {} more cells to hold {positions} positions and the pool has {} \
             of {} free; the request ends here and the others go on",
            least.saturating_sub(held),
            lease.pool.free(),
            lease.pool.total()
        ))
    })?;
    Ok(least)
}

/// Checks the card before session `who` re-lays its planes: the old planes stay until the copy
/// ends, so the new ones (`bytes_of(cells)`) must fit in the free bytes beside them. When they do
/// not, the session retries once at exactly the cells its positions need; when that does not fit
/// either it is refused, naming the bytes it needs and the card has, and its lease goes back to
/// `held`. `free` is `None` when the card cannot be read: the allocation decides then. Returns
/// the cells the lease now holds.
pub(crate) fn fit_planes(
    lease: &mut CellLease,
    who: impl std::fmt::Display,
    held: usize,
    cells: usize,
    positions: usize,
    free: Option<u64>,
    bytes_of: impl Fn(usize) -> u64,
) -> Result<usize, EngError> {
    let Some(free) = free else {
        return Ok(cells);
    };
    if free >= bytes_of(cells) {
        return Ok(cells);
    }
    if positions < cells && free >= bytes_of(positions) {
        lease.resize(positions);
        return Ok(positions);
    }
    lease.resize(held);
    Err(EngError::MemoryPlan(format!(
        "session {who} cannot grow from {held} to {cells} cells: the new planes need {} B beside \
         the old ones until the copy ends and the card has {free} B free; the request ends here \
         and the others go on",
        bytes_of(cells)
    )))
}

/// The cells of an even share of a pool of `pool` cells among `sessions`, whole granules.
pub(crate) fn even_share(pool: usize, sessions: usize) -> usize {
    (pool / sessions.max(1) / CELL_GRANULE * CELL_GRANULE).max(INITIAL_CELLS)
}

fn check_cells(cells: usize, ceiling: usize) -> Result<(), EngError> {
    if cells == 0 || cells > ceiling {
        return Err(EngError::BadRequest(format!(
            "a session holds 1 to {ceiling} cells, got {cells}"
        )));
    }
    Ok(())
}

/// The cells a request must be able to generate into, at least, before it is admitted.
const MIN_COMPLETION_CELLS: usize = 256;

/// The cells of one executor, shared by its sessions.
pub(crate) struct CellPool {
    total: usize,
    used: AtomicUsize,
}

impl CellPool {
    pub(crate) fn new(total: usize) -> Arc<Self> {
        Arc::new(Self {
            total,
            used: AtomicUsize::new(0),
        })
    }

    pub(crate) fn total(&self) -> usize {
        self.total
    }

    /// The cells no session holds.
    pub(crate) fn free(&self) -> usize {
        self.total.saturating_sub(self.used.load(Ordering::Acquire))
    }

    /// A lease of `cells` for a session of at most `ceiling`: a memory-plan error when the pool
    /// holds fewer free, so the request queues.
    pub(crate) fn lease_for(
        self: &Arc<Self>,
        cells: usize,
        ceiling: usize,
    ) -> Result<CellLease, EngError> {
        check_cells(cells, ceiling)?;
        self.lease(cells).ok_or_else(|| {
            EngError::MemoryPlan(format!(
                "the cache pool has {} cells free, {cells} are needed",
                self.free()
            ))
        })
    }

    /// Takes `cells` for a new session, or `None` when the pool holds fewer free.
    pub(crate) fn lease(self: &Arc<Self>, cells: usize) -> Option<CellLease> {
        let mut used = self.used.load(Ordering::Acquire);
        loop {
            let next = used.checked_add(cells).filter(|&next| next <= self.total)?;
            match self
                .used
                .compare_exchange(used, next, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => {
                    return Some(CellLease {
                        pool: Arc::clone(self),
                        cells,
                    });
                }
                Err(seen) => used = seen,
            }
        }
    }
}

/// One session's cells; they return to the pool when the lease drops.
pub(crate) struct CellLease {
    pool: Arc<CellPool>,
    cells: usize,
}

impl CellLease {
    pub(crate) fn cells(&self) -> usize {
        self.cells
    }

    /// The most this lease can be resized to: its own cells and every free one.
    pub(crate) fn reachable(&self) -> usize {
        self.cells + self.pool.free()
    }

    /// [`CellLease::resize`] for a session of at most `ceiling`: a memory-plan error and no change
    /// when the pool cannot fund the growth.
    pub(crate) fn resize_for(&mut self, cells: usize, ceiling: usize) -> Result<(), EngError> {
        check_cells(cells, ceiling)?;
        let more = cells.saturating_sub(self.cells);
        if self.resize(cells) {
            return Ok(());
        }
        Err(EngError::MemoryPlan(format!(
            "the cache pool has {} cells free, {more} more are needed",
            self.pool.free()
        )))
    }

    /// Changes the lease to `cells`; `false` and no change when the pool cannot fund the growth.
    pub(crate) fn resize(&mut self, cells: usize) -> bool {
        if cells <= self.cells {
            self.pool
                .used
                .fetch_sub(self.cells - cells, Ordering::AcqRel);
            self.cells = cells;
            return true;
        }
        let growth = cells - self.cells;
        let mut used = self.pool.used.load(Ordering::Acquire);
        loop {
            let Some(next) = used
                .checked_add(growth)
                .filter(|&next| next <= self.pool.total)
            else {
                return false;
            };
            match self
                .pool
                .used
                .compare_exchange(used, next, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => {
                    self.cells = cells;
                    return true;
                }
                Err(seen) => used = seen,
            }
        }
    }
}

impl Drop for CellLease {
    fn drop(&mut self) {
        self.pool.used.fetch_sub(self.cells, Ordering::AcqRel);
    }
}

/// What the request asks of a session.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CellRequest {
    pub(crate) prompt: usize,
    /// The generation limit; `None` when the request names none.
    pub(crate) completion: Option<u32>,
}

/// What a session's cell count is sized against.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CellRoom {
    /// The cells the session may reach: its own and every free one in the pool.
    pub(crate) reachable: usize,
    /// The most one session carries (the model's context).
    pub(crate) ceiling: usize,
    /// The even share of the pool: what a request that names no limit may use for its whole
    /// context while other sessions keep theirs.
    pub(crate) share: usize,
}

fn round_up(cells: usize) -> usize {
    cells.div_ceil(CELL_GRANULE).saturating_mul(CELL_GRANULE)
}

/// The cells one request's session holds, or `None` when the pool cannot fund the prompt and a
/// minimal generation yet, so the request queues.
///
/// A request that names its generation limit takes the prompt and that limit; one that names
/// none takes the even share, or the prompt and [`OPEN_COMPLETION_CELLS`] when that is more, so
/// a prompt above an even share is never refused while the pool can hold it. The count is
/// rounded up to [`CELL_GRANULE`], capped by the ceiling, and cut to what the pool can reach.
pub(crate) fn session_cells(request: CellRequest, room: CellRoom) -> Option<usize> {
    let completion = request.completion.map(|limit| limit as usize);
    let wanted = match completion {
        Some(limit) => request.prompt.saturating_add(limit.max(1)),
        None => room
            .share
            .max(request.prompt.saturating_add(OPEN_COMPLETION_CELLS)),
    };
    let wanted = round_up(wanted).min(room.ceiling);
    if request.prompt >= room.ceiling {
        return None;
    }
    let least = request
        .prompt
        .saturating_add(
            completion
                .unwrap_or(MIN_COMPLETION_CELLS)
                .clamp(1, MIN_COMPLETION_CELLS),
        )
        .min(room.ceiling);
    if room.reachable < least {
        return None;
    }
    Some(wanted.min(room.reachable).max(least))
}

/// [`session_cells`] as a refusal: a memory-plan error when the pool cannot fund the request yet.
pub(crate) fn session_cells_or_queue(
    request: CellRequest,
    room: CellRoom,
) -> Result<usize, EngError> {
    session_cells(request, room).ok_or_else(|| {
        EngError::MemoryPlan(format!(
            "the cache pool holds {} cells for this session; a {} token prompt and its generation \
             need more",
            room.reachable, request.prompt
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_session_grows_by_half_in_whole_granules_up_to_its_ceiling() {
        assert_eq!(grow_target(1024, 1025, 65_536), 2048);
        assert_eq!(grow_target(4096, 4097, 65_536), 6144);
        assert_eq!(grow_target(60_000, 60_001, 65_536), 65_536);
        // A need beyond half again is met, rounded up.
        assert_eq!(grow_target(1024, 5000, 65_536), 5120);
        // Never fewer than the need, never more than the ceiling.
        assert_eq!(grow_target(1024, 1025, 1025), 1025);
        assert_eq!(grow_target(1024, 1025, 1500), 1500);
        for current in [1024, 3072, 40_960] {
            let next = grow_target(current, current + 1, 262_144);
            assert!(next > current && next.is_multiple_of(CELL_GRANULE));
        }
    }

    const CEILING: usize = 65_536;

    fn room(reachable: usize, share: usize) -> CellRoom {
        CellRoom {
            reachable,
            ceiling: CEILING,
            share,
        }
    }

    #[test]
    fn the_pool_funds_leases_until_it_is_spent_and_takes_cells_back_on_drop() {
        let pool = CellPool::new(10_000);
        let mut first = pool.lease(6_000).expect("first");
        assert_eq!(pool.free(), 4_000);
        assert!(pool.lease(4_001).is_none());
        let second = pool.lease(4_000).expect("second");
        assert_eq!(pool.free(), 0);
        assert!(!first.resize(7_000), "no cell is free to grow into");
        assert_eq!(first.cells(), 6_000);
        drop(second);
        assert!(first.resize(9_000));
        assert_eq!(pool.free(), 1_000);
        assert!(first.resize(2_000));
        assert_eq!(pool.free(), 8_000);
        assert_eq!(first.reachable(), 10_000);
        drop(first);
        assert_eq!(pool.free(), 10_000);
    }

    #[test]
    fn a_prompt_above_the_even_share_takes_what_it_needs_while_the_pool_holds_it() {
        // Eight even shares of 61,440 cells: a 64K prompt is above one share.
        let share = 61_440;
        let lone = session_cells(
            CellRequest {
                prompt: 63_000,
                completion: None,
            },
            room(8 * share, share),
        );
        assert_eq!(lone, Some(CEILING));
        // The same prompt while seven shares are held: the pool still holds it.
        let busy = session_cells(
            CellRequest {
                prompt: 63_000,
                completion: None,
            },
            room(share + 4_000, share),
        );
        assert_eq!(busy, Some(share + 4_000));
        // Not while the pool cannot hold the prompt and a minimal generation: it queues.
        let queued = session_cells(
            CellRequest {
                prompt: 63_000,
                completion: None,
            },
            room(63_100, share),
        );
        assert_eq!(queued, None);
    }

    #[test]
    fn short_requests_take_their_even_share_and_eight_of_them_fit_the_pool() {
        let share = 61_440;
        let pool = CellPool::new(8 * share);
        let leases: Vec<_> = (0..8)
            .map(|_| {
                let cells = session_cells(
                    CellRequest {
                        prompt: 40,
                        completion: None,
                    },
                    room(pool.free(), share),
                )
                .expect("cells");
                assert_eq!(cells, share);
                pool.lease(cells).expect("lease")
            })
            .collect();
        assert_eq!(pool.free(), 0);
        drop(leases);
    }

    #[test]
    fn a_request_with_a_limit_takes_the_prompt_and_the_limit() {
        let cells = session_cells(
            CellRequest {
                prompt: 40,
                completion: Some(200),
            },
            room(1 << 20, 61_440),
        );
        assert_eq!(cells, Some(1024));
        let cells = session_cells(
            CellRequest {
                prompt: 3_000,
                completion: Some(2_000),
            },
            room(1 << 20, 61_440),
        );
        assert_eq!(cells, Some(5_120));
    }

    #[test]
    fn the_ceiling_bounds_a_session_and_a_prompt_that_cannot_generate_queues_nowhere() {
        let cells = session_cells(
            CellRequest {
                prompt: 10,
                completion: Some(10_000_000),
            },
            room(1 << 30, 61_440),
        );
        assert_eq!(cells, Some(CEILING));
        // No cell is left to generate into: the request is not a queue candidate.
        let none = session_cells(
            CellRequest {
                prompt: CEILING,
                completion: None,
            },
            room(1 << 30, 61_440),
        );
        assert_eq!(none, None);
    }

    #[test]
    fn a_session_is_never_given_fewer_cells_than_its_prompt_and_a_minimal_generation() {
        for prompt in [1usize, 100, 1_023, 1_024, 30_000, 65_000] {
            for completion in [None, Some(0u32), Some(1), Some(5_000)] {
                for reachable in [prompt, prompt + 255, prompt + 256, 80_000, 1 << 20] {
                    let got =
                        session_cells(CellRequest { prompt, completion }, room(reachable, 8_192));
                    if let Some(cells) = got {
                        assert!(cells > prompt, "{prompt} {completion:?} {reachable}");
                        assert!(cells <= reachable.min(CEILING));
                    }
                }
            }
        }
    }
}
