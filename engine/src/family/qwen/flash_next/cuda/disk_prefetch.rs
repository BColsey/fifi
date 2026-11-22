//! Bounded speculative reads with optional reuse of an exact match already in flight.

use crate::EngError;
use std::{
    collections::VecDeque,
    sync::{Arc, Condvar, Mutex, mpsc},
    thread::{Builder, JoinHandle},
    time::{Duration, Instant},
};

pub(super) const CANDIDATES: usize = 4;
const HANDOFF_GRACE: Duration = Duration::from_micros(500);

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct Target {
    pub(super) position: u64,
    pub(super) layer: usize,
}

struct Plan {
    target: Target,
    generation: u64,
    pending: VecDeque<u16>,
    ready: Vec<(u16, Vec<u8>)>,
}

#[derive(Default)]
struct Counters {
    issued: u64,
    failed: u64,
    ready_hits: u64,
    late: u64,
    demand: u64,
    joined: u64,
    join_fallback: u64,
    join_wait_ns: u64,
    last_error: Option<String>,
}

struct ActiveRead {
    generation: u64,
    expert: u16,
    handoff: Option<mpsc::Sender<Vec<u8>>>,
}

#[derive(Default)]
struct State {
    generation: u64,
    plans: VecDeque<Plan>,
    demand_active: bool,
    active: Option<ActiveRead>,
    stop: bool,
    counters: Counters,
}

#[derive(Default)]
struct Shared {
    state: Mutex<State>,
    wake: Condvar,
}

pub(super) struct DiskPrefetch {
    horizon: usize,
    join_active: bool,
    shared: Arc<Shared>,
    worker: Option<JoinHandle<()>>,
}

impl DiskPrefetch {
    pub(super) fn spawn(
        horizon: usize,
        join_active: bool,
        mut read: impl FnMut(usize, u16) -> Result<Vec<u8>, EngError> + Send + 'static,
    ) -> Result<Self, EngError> {
        let shared = Arc::new(Shared::default());
        let work = Arc::clone(&shared);
        let worker = Builder::new()
            .name("flash-next-disk-prefetch".into())
            .spawn(move || {
                while let Some((generation, layer, expert)) = work.next() {
                    let record = read(layer, expert);
                    let mut state = work.state.lock().expect("disk prefetch state");
                    let active = state.active.take().expect("one active speculative read");
                    match record {
                        Ok(bytes) => {
                            if let Some(sender) = active.handoff {
                                let _ = sender.send(bytes);
                            } else if let Some(plan) =
                                state.plans.iter_mut().find(|p| p.generation == generation)
                            {
                                plan.ready.push((expert, bytes));
                            }
                        }
                        Err(error) => {
                            state.counters.failed += 1;
                            state.counters.last_error = Some(error.describe());
                        }
                    }
                }
            })
            .map_err(|error| EngError::Backend {
                operation: "Flash-Next disk prefetch worker",
                message: error.to_string(),
            })?;
        Ok(Self {
            horizon,
            join_active,
            shared,
            worker: Some(worker),
        })
    }

    pub(super) fn horizon(&self) -> usize {
        self.horizon
    }

    pub(super) fn submit(&self, target: Target, experts: impl Iterator<Item = u16>) {
        let pending: VecDeque<_> = experts.take(CANDIDATES).collect();
        if pending.is_empty() {
            return;
        }
        let mut state = self.shared.state.lock().expect("disk prefetch state");
        state.generation += 1;
        let generation = state.generation;
        state
            .plans
            .retain(|plan| plan.target.position == target.position && plan.target != target);
        state.plans.push_back(Plan {
            target,
            generation,
            pending,
            ready: Vec::new(),
        });
        state
            .plans
            .make_contiguous()
            .sort_unstable_by_key(|plan| plan.target.layer);
        state.plans.truncate(self.horizon);
        self.shared.wake.notify_one();
    }

    pub(super) fn cancel(&self) {
        let mut state = self.shared.state.lock().expect("disk prefetch state");
        state.plans.clear();
    }

    /// Takes the current layer's completed records, dropping any other predictions
    /// for that layer. Future layers stay queued, with launches paused during demand.
    /// The single read already in flight uses its own slot and may finish in parallel.
    pub(super) fn read_demand(
        &self,
        target: Target,
        experts: &[u16],
        mut read: impl FnMut(&[u16]) -> Result<Vec<Vec<u8>>, EngError>,
    ) -> Result<Vec<Vec<u8>>, EngError> {
        let (mut ready, handoff) = {
            let mut state = self.shared.state.lock().expect("disk prefetch state");
            state.demand_active = true;
            let plan = state
                .plans
                .iter()
                .position(|plan| plan.target == target)
                .and_then(|i| state.plans.remove(i));
            state.plans.retain(|plan| {
                plan.target.position == target.position && plan.target.layer > target.layer
            });
            let mut handoff = None;
            if let Some(plan) = &plan
                && let Some(active) = &mut state.active
                && active.generation == plan.generation
                && let Some(index) = experts.iter().position(|&id| id == active.expert)
            {
                if self.join_active {
                    let (sender, receiver) = mpsc::channel();
                    active.handoff = Some(sender);
                    handoff = Some((index, receiver));
                }
                state.counters.late += 1;
            }
            (plan.map_or_else(Vec::new, |plan| plan.ready), handoff)
        };
        let held_index = handoff.as_ref().map(|(index, _)| *index);
        let mut missing = Vec::new();
        let mut ready_hits = 0;
        let mut ordered: Vec<_> = experts
            .iter()
            .enumerate()
            .map(|(position, &expert)| {
                if Some(position) == held_index {
                    None
                } else if let Some(index) = ready.iter().position(|&(id, _)| id == expert) {
                    ready_hits += 1;
                    Some(ready.swap_remove(index).1)
                } else {
                    missing.push(expert);
                    None
                }
            })
            .collect();
        {
            let mut state = self.shared.state.lock().expect("disk prefetch state");
            state.counters.demand += missing.len() as u64;
            state.counters.ready_hits += ready_hits;
        }

        drop(ready);
        // Dispatch the rest of the exact misses first. The adopted read gets
        // only a short grace period after that batch returns.
        let result = (|| {
            let mut fetched = read(&missing)?.into_iter();
            if let Some((index, receiver)) = handoff {
                let started = Instant::now();
                let bytes = receiver.recv_timeout(HANDOFF_GRACE).ok();
                {
                    let mut state = self.shared.state.lock().expect("disk prefetch state");
                    state.counters.join_wait_ns += started.elapsed().as_nanos() as u64;
                    if bytes.is_some() {
                        state.counters.joined += 1;
                    } else {
                        state.counters.join_fallback += 1;
                        state.counters.demand += 1;
                    }
                }
                ordered[index] = Some(match bytes {
                    Some(bytes) => bytes,
                    None => read(&[experts[index]])?
                        .pop()
                        .expect("one requested record"),
                });
            }
            Ok(ordered
                .into_iter()
                .map(|record| {
                    record.unwrap_or_else(|| fetched.next().expect("one record per miss"))
                })
                .collect())
        })();
        self.shared
            .state
            .lock()
            .expect("disk prefetch state")
            .demand_active = false;
        self.shared.wake.notify_one();
        result
    }

    pub(super) fn report(&self) -> String {
        let state = self.shared.state.lock().expect("disk prefetch state");
        let stats = &state.counters;
        format!(
            "disk prefetch cumulative: issued {} failed {} ready hits {} late {} demand {} active {} joined {} join fallback {} join wait ns {}{}",
            stats.issued,
            stats.failed,
            stats.ready_hits,
            stats.late,
            stats.demand,
            usize::from(state.active.is_some()),
            stats.joined,
            stats.join_fallback,
            stats.join_wait_ns,
            stats
                .last_error
                .as_ref()
                .map_or(String::new(), |error| format!(" last error: {error}"))
        )
    }
}

impl Shared {
    fn next(&self) -> Option<(u64, usize, u16)> {
        let mut state = self.state.lock().expect("disk prefetch state");
        loop {
            if state.stop {
                return None;
            }
            if !state.demand_active
                && let Some(plan) = state.plans.iter_mut().find(|plan| !plan.pending.is_empty())
                && let Some(expert) = plan.pending.pop_front()
            {
                let layer = plan.target.layer;
                let generation = plan.generation;
                state.active = Some(ActiveRead {
                    generation,
                    expert,
                    handoff: None,
                });
                state.counters.issued += 1;
                return Some((generation, layer, expert));
            }
            state = self.wake.wait(state).expect("disk prefetch state");
        }
    }
}

impl Drop for DiskPrefetch {
    fn drop(&mut self) {
        self.shared.state.lock().expect("disk prefetch state").stop = true;
        self.shared.wake.notify_one();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;
    use std::time::{Duration, Instant};

    fn target() -> Target {
        Target {
            position: 7,
            layer: 5,
        }
    }

    #[test]
    fn ready_records_replace_reads_and_keep_exact_order() {
        let prefetch = DiskPrefetch::spawn(1, false, |layer, id| {
            assert_eq!(layer, 5);
            Ok(vec![id as u8])
        })
        .unwrap();
        prefetch.submit(target(), [4, 3].into_iter());
        let deadline = Instant::now() + Duration::from_secs(3);
        loop {
            let state = prefetch.shared.state.lock().unwrap();
            if state.plans.front().unwrap().ready.len() == 2 {
                break;
            }
            drop(state);
            assert!(Instant::now() < deadline);
            std::thread::yield_now();
        }
        let bytes = prefetch
            .read_demand(target(), &[9, 3, 8, 4], |ids| {
                assert_eq!(ids, [9, 8]);
                Ok(ids.iter().map(|&id| vec![id as u8]).collect())
            })
            .unwrap();
        assert_eq!(bytes, [vec![9], vec![3], vec![8], vec![4]]);
        assert!(
            prefetch
                .report()
                .contains("issued 2 failed 0 ready hits 2 late 0 demand 2")
        );
    }

    #[test]
    fn demand_skips_an_inflight_guess_and_cancel_discards_its_result() {
        let (entered, started) = mpsc::channel();
        let (release, proceed) = mpsc::channel();
        let prefetch = DiskPrefetch::spawn(1, false, move |_, id| {
            entered.send(id).unwrap();
            proceed.recv_timeout(Duration::from_secs(3)).unwrap();
            Ok(vec![id as u8])
        })
        .unwrap();
        prefetch.submit(target(), [3, 4].into_iter());
        assert_eq!(started.recv_timeout(Duration::from_secs(3)).unwrap(), 3);
        let demand = prefetch
            .read_demand(target(), &[3], |ids| {
                assert_eq!(ids, [3]);
                Ok(vec![vec![30]])
            })
            .unwrap();
        assert_eq!(demand, [vec![30]]);
        assert!(
            prefetch
                .report()
                .contains("issued 1 failed 0 ready hits 0 late 1 demand 1 active 1")
        );
        prefetch.submit(target(), [3].into_iter());
        prefetch.cancel();
        release.send(()).unwrap();
        let deadline = Instant::now() + Duration::from_secs(3);
        while prefetch.shared.state.lock().unwrap().active.is_some() {
            assert!(Instant::now() < deadline);
            std::thread::yield_now();
        }
        let demand = prefetch
            .read_demand(target(), &[3], |ids| {
                assert_eq!(ids, [3]);
                Ok(vec![vec![31]])
            })
            .unwrap();
        assert_eq!(demand, [vec![31]]);
        assert!(
            prefetch
                .report()
                .contains("issued 1 failed 0 ready hits 0 late 1 demand 2")
        );
    }

    #[test]
    fn demand_pauses_queued_reads_without_discarding_future_layers() {
        let (entered, started) = mpsc::channel();
        let (release, proceed) = mpsc::channel();
        let prefetch = DiskPrefetch::spawn(2, false, move |layer, id| {
            entered.send((layer, id)).unwrap();
            proceed.recv_timeout(Duration::from_secs(3)).unwrap();
            Ok(vec![id as u8])
        })
        .unwrap();
        let first = target();
        let second = Target { layer: 6, ..first };
        prefetch.submit(first, [3, 4].into_iter());
        assert_eq!(
            started.recv_timeout(Duration::from_secs(3)).unwrap(),
            (5, 3)
        );
        prefetch.submit(second, [9].into_iter());

        let intervening = Target { layer: 4, ..first };
        let demand = prefetch
            .read_demand(intervening, &[7], |ids| {
                assert_eq!(ids, [7]);
                release.send(()).unwrap();
                let deadline = Instant::now() + Duration::from_secs(3);
                while prefetch.shared.state.lock().unwrap().active.is_some() {
                    assert!(Instant::now() < deadline);
                    std::thread::yield_now();
                }
                assert!(started.try_recv().is_err());
                Ok(vec![vec![70]])
            })
            .unwrap();
        assert_eq!(demand, [vec![70]]);

        for next in [(5, 4), (6, 9)] {
            assert_eq!(started.recv_timeout(Duration::from_secs(3)).unwrap(), next);
            release.send(()).unwrap();
        }
        let deadline = Instant::now() + Duration::from_secs(3);
        while prefetch.shared.state.lock().unwrap().active.is_some() {
            assert!(Instant::now() < deadline);
            std::thread::yield_now();
        }
        assert_eq!(
            prefetch
                .read_demand(first, &[4, 3], |ids| {
                    assert!(ids.is_empty());
                    Ok(Vec::new())
                })
                .unwrap(),
            [vec![4], vec![3]]
        );
        assert_eq!(
            prefetch
                .read_demand(second, &[9], |ids| {
                    assert!(ids.is_empty());
                    Ok(Vec::new())
                })
                .unwrap(),
            [vec![9]]
        );
    }

    #[test]
    fn matching_inflight_read_is_reused_after_dispatching_the_other_exact_misses() {
        let (entered, started) = mpsc::channel();
        let (release, proceed) = mpsc::channel();
        let prefetch = DiskPrefetch::spawn(2, true, move |_, id| {
            entered.send(id).unwrap();
            proceed.recv_timeout(Duration::from_secs(3)).unwrap();
            Ok(vec![id as u8])
        })
        .unwrap();
        prefetch.submit(target(), [3, 4].into_iter());
        assert_eq!(started.recv_timeout(Duration::from_secs(3)).unwrap(), 3);
        let result = prefetch
            .read_demand(target(), &[9, 3, 8], |ids| {
                assert_eq!(ids, [9, 8]);
                release.send(()).unwrap();
                let deadline = Instant::now() + Duration::from_secs(3);
                while prefetch.shared.state.lock().unwrap().active.is_some() {
                    assert!(Instant::now() < deadline);
                    std::thread::yield_now();
                }
                Ok(vec![vec![90], vec![80]])
            })
            .unwrap();
        assert_eq!(result, [vec![90], vec![3], vec![80]]);
        assert!(
            prefetch
                .report()
                .contains("late 1 demand 2 active 0 joined 1 join fallback 0")
        );
    }

    #[test]
    fn stalled_inflight_read_falls_back_to_the_exact_reader() {
        let (entered, started) = mpsc::channel();
        let (release, proceed) = mpsc::channel();
        let prefetch = DiskPrefetch::spawn(2, true, move |_, id| {
            entered.send(id).unwrap();
            proceed.recv_timeout(Duration::from_secs(3)).unwrap();
            Ok(vec![id as u8])
        })
        .unwrap();
        prefetch.submit(target(), [3, 4].into_iter());
        assert_eq!(started.recv_timeout(Duration::from_secs(3)).unwrap(), 3);
        let mut calls = Vec::new();
        let result = prefetch
            .read_demand(target(), &[3], |ids| {
                calls.push(ids.to_vec());
                Ok(ids.iter().map(|_| vec![30]).collect())
            })
            .unwrap();
        assert_eq!(result, [vec![30]]);
        assert_eq!(calls, [vec![], vec![3]]);
        assert!(
            prefetch
                .report()
                .contains("late 1 demand 1 active 1 joined 0 join fallback 1")
        );
        release.send(()).unwrap();
    }

    #[test]
    fn a_failed_inflight_read_falls_back_to_the_exact_reader_and_keeps_its_error() {
        let (entered, started) = mpsc::channel();
        let (release, proceed) = mpsc::channel();
        let prefetch = DiskPrefetch::spawn(2, true, move |_, id| {
            entered.send(id).unwrap();
            proceed.recv_timeout(Duration::from_secs(3)).unwrap();
            Err(EngError::BadRequest(format!("record {id} unreadable")))
        })
        .unwrap();
        prefetch.submit(target(), [3, 4].into_iter());
        assert_eq!(started.recv_timeout(Duration::from_secs(3)).unwrap(), 3);
        let mut calls = Vec::new();
        let result = prefetch
            .read_demand(target(), &[3], |ids| {
                calls.push(ids.to_vec());
                if ids.is_empty() {
                    // Let the adopted read fail and drop its hand-off before the wait.
                    release.send(()).unwrap();
                    let deadline = Instant::now() + Duration::from_secs(3);
                    while prefetch.shared.state.lock().unwrap().active.is_some() {
                        assert!(Instant::now() < deadline);
                        std::thread::yield_now();
                    }
                }
                Ok(ids.iter().map(|_| vec![30]).collect())
            })
            .unwrap();
        assert_eq!(result, [vec![30]]);
        assert_eq!(calls, [vec![], vec![3]]);
        let report = prefetch.report();
        assert!(
            report.contains("late 1 demand 1 active 0 joined 0 join fallback 1"),
            "{report}"
        );
        assert!(report.contains("failed 1"), "{report}");
        assert!(report.contains("last error: "), "{report}");
        assert!(report.contains("record 3 unreadable"), "{report}");
    }
}
