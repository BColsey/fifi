//! One layer of speculative disk reads. Demand consumes only records already packed in RAM.

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
    plan: Option<Plan>,
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
    shared: Arc<Shared>,
    worker: Option<JoinHandle<()>>,
}

impl DiskPrefetch {
    pub(super) fn spawn(
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
                        Ok(bytes) if state.generation == generation => {
                            if let Some(plan) = &mut state.plan {
                                plan.ready.push((expert, bytes));
                            }
                        }
                        Err(_) => state.counters.failed += 1,
                        _ => {}
                    }
                }
            })
            .map_err(|error| EngError::Backend {
                operation: "Flash-Next disk prefetch worker",
                message: error.to_string(),
            })?;
        Ok(Self {
            shared,
            worker: Some(worker),
        })
    }

    pub(super) fn submit(&self, target: Target, experts: impl Iterator<Item = u16>) {
        let pending: VecDeque<_> = experts.take(CANDIDATES).collect();
        if pending.is_empty() {
            return;
        }
        let mut state = self.shared.state.lock().expect("disk prefetch state");
        state.generation += 1;
        state.plan = Some(Plan {
            target,
            pending,
            ready: Vec::new(),
        });
        self.shared.wake.notify_one();
    }

    pub(super) fn cancel(&self) {
        let mut state = self.shared.state.lock().expect("disk prefetch state");
        state.generation += 1;
        state.plan = None;
    }

    /// Closes the speculative queue before demand dispatch. A read already in the
    /// kernel may finish, but it owns a separate slot and never delays demand on a lock.
    pub(super) fn read_demand(
        &self,
        target: Target,
        experts: &[u16],
        read: impl FnOnce(&[u16]) -> Result<Vec<Vec<u8>>, EngError>,
    ) -> Result<Vec<Vec<u8>>, EngError> {
        let mut ready = {
            let mut state = self.shared.state.lock().expect("disk prefetch state");
            let plan = state.plan.take().filter(|plan| plan.target == target);
            if plan.is_some()
                && let Some((generation, expert)) = state.active
                && generation == state.generation
                && experts.contains(&expert)
            {
                state.counters.late += 1;
            }
            state.generation += 1;
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
        let mut fetched = read(&missing)?.into_iter();
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
            if let Some(plan) = &mut state.plan
                && let Some(expert) = plan.pending.pop_front()
            {
                let layer = plan.target.layer;
                let generation = state.generation;
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
        let prefetch = DiskPrefetch::spawn(|layer, id| {
            assert_eq!(layer, 5);
            Ok(vec![id as u8])
        })
        .unwrap();
        prefetch.submit(target(), [4, 3].into_iter());
        let deadline = Instant::now() + Duration::from_secs(3);
        loop {
            let state = prefetch.shared.state.lock().unwrap();
            if state.plan.as_ref().unwrap().ready.len() == 2 {
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
        let prefetch = DiskPrefetch::spawn(move |_, id| {
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
}
