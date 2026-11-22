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
