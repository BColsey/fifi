#![allow(unsafe_code)]
//! Persistent worker threads for the EXL3 CPU expert kernels.
//!
//! WHAT: [`Exl3Pool::run`] executes `chunks` independent calls of one closure, `f(0)` to
//! `f(chunks - 1)`, across the calling thread and `threads - 1` persistent workers, and returns
//! when the last one finished. Chunks are claimed from one atomic counter, so a descheduled
//! worker only delays the chunk it holds.
//!
//! WHY a pool of its own: the kernels run several short phases per MoE layer (rotate, project,
//! activate, project, rotate), each a few microseconds to tens of microseconds long, so a phase
//! cannot pay a thread spawn, and the hand-off must be a spin on an epoch counter. The NVFP4
//! expert pool in `nvfp4_cpu` has the same wake-up design, but its job record carries that
//! kernel's two-phase chunk schedule; this one hands over a plain closure. Idle workers wait
//! per the shared [`WaitPolicy`].
//!
//! Unsafe scope: one lifetime-erased closure pointer. `run` does not return before every claimed
//! chunk finished, so the pointee outlives every dereference. Confined to this file by the
//! `unsafe_is_confined_to_the_avx2_module` test.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::EngError;
use crate::kernels::nvfp4_cpu::WaitPolicy;

type ChunkFn<'a> = dyn Fn(usize) + Sync + 'a;

#[repr(align(64))]
struct Padded<T>(T);

/// One call as the workers see it.
struct Job {
    ctx: *const ChunkFn<'static>,
    total: usize,
    next: Padded<AtomicUsize>,
    done: Padded<AtomicUsize>,
    panicked: AtomicBool,
}

// SAFETY: `ctx` is dereferenced only after a successful claim of a chunk index below `total`,
// SAFETY: and `run` does not return before `done == total`, so the closure outlives every
// SAFETY: dereference. The other fields are atomics or immutable.
unsafe impl Send for Job {}
// SAFETY: see above; the closure is `Sync`, all other shared mutation goes through atomics.
unsafe impl Sync for Job {}

impl Job {
    fn work(&self) {
        loop {
            let i = self.next.0.fetch_add(1, Ordering::Relaxed);
            if i >= self.total {
                return;
            }
            // SAFETY: chunk `i < total` was claimed above, so the closure is alive (see Send).
            let result = catch_unwind(AssertUnwindSafe(|| unsafe { (*self.ctx)(i) }));
            if result.is_err() {
                self.panicked.store(true, Ordering::Relaxed);
            }
            self.done.0.fetch_add(1, Ordering::Release);
        }
    }
}

struct Slot {
    epoch: u64,
    job: Option<Arc<Job>>,
    parked: usize,
    shutdown: bool,
}

struct Shared {
    slot: Mutex<Slot>,
    wake: Condvar,
    /// Mirrors `Slot::epoch` so spinning workers poll without the lock.
    epoch: AtomicU64,
    wait: WaitPolicy,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

impl Shared {
    fn publish(&self, job: Option<Arc<Job>>, shutdown: bool) {
        let mut slot = lock(&self.slot);
        slot.epoch += 1;
        slot.job = job;
        slot.shutdown = shutdown;
        self.epoch.store(slot.epoch, Ordering::Release);
        let parked = slot.parked;
        drop(slot);
        if parked > 0 {
            self.wake.notify_all();
        }
    }

    fn spin_window(&self) -> Option<Duration> {
        match self.wait {
            WaitPolicy::Spin => Some(Duration::MAX),
            WaitPolicy::Park => None,
            WaitPolicy::SpinThenPark(window) => Some(window),
        }
    }

    /// Busy-waits for a new epoch within the policy window. The spin yields the CPU now and
    /// then, so workers cannot starve the caller when threads outnumber the available cores.
    fn spin_for_epoch(&self, seen: u64) {
        let Some(window) = self.spin_window() else {
            return;
        };
        let start = Instant::now();
        let mut spins = 0u32;
        while self.epoch.load(Ordering::Acquire) == seen {
            spins = spins.wrapping_add(1);
            if spins.is_multiple_of(64) && start.elapsed() >= window {
                return;
            }
            if spins.is_multiple_of(4096) {
                std::thread::yield_now();
            }
            std::hint::spin_loop();
        }
    }

    fn next_job(&self, seen: u64) -> (u64, Option<Arc<Job>>, bool) {
        self.spin_for_epoch(seen);
        let mut slot = lock(&self.slot);
        while slot.epoch == seen {
            slot.parked += 1;
            slot = self.wake.wait(slot).unwrap_or_else(PoisonError::into_inner);
            slot.parked -= 1;
        }
        (slot.epoch, slot.job.clone(), slot.shutdown)
    }
}

fn worker_main(shared: Arc<Shared>) {
    let mut seen = 0u64;
    loop {
        let (epoch, job, shutdown) = shared.next_job(seen);
        if shutdown {
            return;
        }
        seen = epoch;
        if let Some(job) = job {
            job.work();
        }
    }
}

/// A persistent set of worker threads; dropping it joins them.
pub struct Exl3Pool {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
    threads: usize,
    /// One call at a time owns the slot.
    call: Mutex<()>,
}

impl Exl3Pool {
    /// `threads` counts the calling thread, so `threads - 1` workers are spawned.
    pub fn new(threads: usize, wait: WaitPolicy) -> Result<Self, EngError> {
        let threads = threads.max(1);
        let shared = Arc::new(Shared {
            slot: Mutex::new(Slot {
                epoch: 0,
                job: None,
                parked: 0,
                shutdown: false,
            }),
            wake: Condvar::new(),
            epoch: AtomicU64::new(0),
            wait,
        });
        let mut pool = Self {
            shared: Arc::clone(&shared),
            workers: Vec::with_capacity(threads - 1),
            threads,
            call: Mutex::new(()),
        };
        for index in 1..threads {
            let worker_shared = Arc::clone(&shared);
            let spawned = std::thread::Builder::new()
                .name(format!("exl3-cpu-{index}"))
                .spawn(move || worker_main(worker_shared));
            match spawned {
                Ok(handle) => pool.workers.push(handle),
                Err(error) => {
                    return Err(EngError::MemoryPlan(format!(
                        "EXL3 CPU pool: spawn worker {index}: {error}"
                    )));
                }
            }
        }
        Ok(pool)
    }

    /// Threads that compute a call, the caller included.
    pub fn threads(&self) -> usize {
        self.threads
    }

    /// Runs `f(0..chunks)` across the pool and returns when all chunks finished. Chunks must
    /// write disjoint memory. A panic inside a chunk is re-raised here after the others finish.
    pub fn run<F: Fn(usize) + Sync>(&self, chunks: usize, f: F) {
        if chunks == 0 {
            return;
        }
        if self.threads == 1 || chunks == 1 {
            (0..chunks).for_each(&f);
            return;
        }
        let _owner = lock(&self.call);
        let erased: &ChunkFn<'_> = &f;
        // SAFETY: only the lifetime bound of the trait object changes. This function waits for
        // SAFETY: `done == chunks` below before `f` is dropped, so no chunk uses it later.
        let ctx = unsafe {
            std::mem::transmute::<*const ChunkFn<'_>, *const ChunkFn<'static>>(
                erased as *const ChunkFn<'_>,
            )
        };
        let job = Arc::new(Job {
            ctx,
            total: chunks,
            next: Padded(AtomicUsize::new(0)),
            done: Padded(AtomicUsize::new(0)),
            panicked: AtomicBool::new(false),
        });
        self.shared.publish(Some(Arc::clone(&job)), false);
        job.work();
        let mut spins = 0u32;
        while job.done.0.load(Ordering::Acquire) != chunks {
            spins = spins.saturating_add(1);
            if spins < 4096 {
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
        assert!(
            !job.panicked.load(Ordering::Relaxed),
            "an EXL3 CPU kernel chunk panicked"
        );
    }
}

impl Drop for Exl3Pool {
    fn drop(&mut self) {
        self.shared.publish(None, true);
        for handle in self.workers.drain(..) {
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pool(threads: usize) -> Exl3Pool {
        Exl3Pool::new(
            threads,
            WaitPolicy::SpinThenPark(Duration::from_micros(200)),
        )
        .unwrap()
    }

    #[test]
    fn every_chunk_runs_exactly_once_across_repeated_calls() {
        let p = pool(4);
        for chunks in [1usize, 2, 7, 64, 1000] {
            let hits: Vec<AtomicUsize> = (0..chunks).map(|_| AtomicUsize::new(0)).collect();
            for _ in 0..20 {
                p.run(chunks, |i| {
                    hits[i].fetch_add(1, Ordering::Relaxed);
                });
            }
            assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 20));
        }
    }

    #[test]
    fn a_single_thread_pool_runs_inline_and_zero_chunks_return() {
        let p = pool(1);
        let sum = AtomicUsize::new(0);
        p.run(5, |i| {
            sum.fetch_add(i, Ordering::Relaxed);
        });
        p.run(0, |_| unreachable!());
        assert_eq!(sum.load(Ordering::Relaxed), 10);
    }

    #[test]
    fn a_panicking_chunk_surfaces_in_the_caller_and_the_pool_survives() {
        let p = pool(3);
        let result = catch_unwind(AssertUnwindSafe(|| {
            p.run(8, |i| assert_ne!(i, 5, "chunk five fails"));
        }));
        assert!(result.is_err());
        let ran = AtomicUsize::new(0);
        p.run(8, |_| {
            ran.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(ran.load(Ordering::Relaxed), 8);
    }
}
