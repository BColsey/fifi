//! Bounded speculative disk reads. Demand consumes records already packed in RAM.

use crate::EngError;
use std::{
    collections::VecDeque,
    sync::{Arc, Condvar, Mutex},
    thread::{Builder, JoinHandle},
};

pub(super) const CANDIDATES: usize = 4;

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
}

#[derive(Default)]
struct State {
    generation: u64,
    plans: VecDeque<Plan>,
    demand_active: bool,
    active: Option<(u64, u16)>,
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
    shared: Arc<Shared>,
    worker: Option<JoinHandle<()>>,
}

impl DiskPrefetch {
    pub(super) fn spawn(
        horizon: usize,
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
                    state.active = None;
                    match record {
                        Ok(bytes) => {
                            if let Some(plan) =
                                state.plans.iter_mut().find(|p| p.generation == generation)
                            {
                                plan.ready.push((expert, bytes));
                            }
                        }
                        Err(_) => state.counters.failed += 1,
                    }
                }
            })
            .map_err(|error| EngError::Backend {
                operation: "Flash-Next disk prefetch worker",
                message: error.to_string(),
            })?;
        Ok(Self {
            horizon,
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
        read: impl FnOnce(&[u16]) -> Result<Vec<Vec<u8>>, EngError>,
    ) -> Result<Vec<Vec<u8>>, EngError> {
        let mut ready = {
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
            if let Some(plan) = &plan
                && let Some((generation, expert)) = state.active
                && generation == plan.generation
                && experts.contains(&expert)
            {
                state.counters.late += 1;
            }
            plan.map_or_else(Vec::new, |plan| plan.ready)
        };
        let mut missing = Vec::new();
        let ordered: Vec<_> = experts
            .iter()
            .map(|&expert| {
                if let Some(index) = ready.iter().position(|&(id, _)| id == expert) {
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
            state.counters.ready_hits += (experts.len() - missing.len()) as u64;
        }

        drop(ready);
        let fetched = read(&missing);
        self.shared
            .state
            .lock()
            .expect("disk prefetch state")
            .demand_active = false;
        self.shared.wake.notify_one();
        let mut fetched = fetched?.into_iter();
        Ok(ordered
            .into_iter()
            .map(|record| record.unwrap_or_else(|| fetched.next().expect("one record per miss")))
            .collect())
    }

    pub(super) fn report(&self) -> String {
        let state = self.shared.state.lock().expect("disk prefetch state");
        let stats = &state.counters;
        format!(
            "disk prefetch cumulative: issued {} failed {} ready hits {} late {} demand {} active {}",
            stats.issued,
            stats.failed,
            stats.ready_hits,
            stats.late,
            stats.demand,
            usize::from(state.active.is_some())
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
                state.active = Some((generation, expert));
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
        let prefetch = DiskPrefetch::spawn(1, |layer, id| {
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
        let prefetch = DiskPrefetch::spawn(1, move |_, id| {
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
        let prefetch = DiskPrefetch::spawn(2, move |layer, id| {
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
}
