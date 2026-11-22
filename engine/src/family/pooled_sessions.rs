//! Several sessions on one resident step, time sliced.
//!
//! The step's weights, context, stream, activation arena, prefill arena and prefix store are
//! shared. A session owns only its sequence part (the host's `Part`): KV planes, DeltaNet and convolution
//! state, step words and the graph captured over them. The cache cells come from one pool
//! (`family::cell_pool`): a session leases the cells its request needs when the request is
//! admitted, and a request the pool cannot fund yet queues in the worker.
//!
//! The step runs the part of the session that was used last. Using another session swaps the two
//! parts (a handle exchange, no device work); a window of rows runs the rows one after another,
//! each through the step's own single-row decode, so a row's tokens are the tokens the same
//! request gets alone. An engine that serves one session at a time never builds this type.

#![cfg_attr(
    not(feature = "native-cuda"),
    allow(
        dead_code,
        reason = "the CUDA-build executors serve their sessions through it"
    )
)]

use crate::{
    EngError,
    family::{
        cell_pool::{
            CellLease, CellPool, CellRequest, CellRoom, INITIAL_CELLS, admission_room, even_share,
            fit_planes, fund_growth, session_cells_or_queue,
        },
        executor::{BatchPolicy, ExecutorOps, GenerationState},
        sampler::Sampler,
        session::{DecodeSession, Limits, SaveKind, StoreLimits, TokenStepper},
        window::WindowRows,
    },
    sched::job::PrefixReport,
    tokenizer::DecodeBatch,
};
use std::{cell::RefCell, rc::Rc, sync::Arc};

type Result<T> = std::result::Result<T, EngError>;

/// What a session keeps on the engine thread: its cell lease, and its sequence part while the step
/// does not run it.
struct SeqSlot<P> {
    inner: RefCell<SlotInner<P>>,
}

struct SlotInner<P> {
    part: Option<P>,
    lease: CellLease,
}

/// What a resident step gives the sessions executor: the part of it that one sequence owns, and
/// the exchange of that part. The weights, context, stream, arenas and prefix store stay the
/// step's own and shared.
pub(crate) trait PartHost: ExecutorOps<Session = DecodeSession> + TokenStepper {
    /// One sequence's state on the device.
    type Part;
    /// The prefix of the host's log lines.
    const LABEL: &'static str;
    /// Whether this step can serve more than one session.
    fn check_sessions(&self) -> Result<()>;
    /// The sessions the host's planner funded beside its other allocations, when it planned them:
    /// the executor never admits more. `None` leaves the count to the caller.
    fn funded_sessions(&self) -> Option<usize> {
        None
    }
    /// The cells of the context the step was built for, which become the pool.
    fn context_cells(&self) -> usize;
    /// A zeroed part holding `cells` positions.
    fn new_part(&mut self, cells: usize) -> Result<Self::Part>;
    /// Exchanges the part the step runs with `part`.
    fn swap_part(&mut self, part: &mut Self::Part);
    /// Gives `part` room for `cells` positions, keeping everything it holds: the sequence goes
    /// on where it was, in larger planes. `part` is out of the step. On an error the part is as
    /// it was.
    fn grow_part(&mut self, part: &mut Self::Part, cells: usize) -> Result<()>;
    /// The cells `part` holds.
    fn part_cells(part: &Self::Part) -> usize;
    /// The positions `part` has consumed.
    fn part_fed(part: &Self::Part) -> usize;
    /// Device bytes of a part holding `cells` positions.
    fn part_bytes_at(&self, cells: usize) -> u64;
    /// Free device memory now, when the card can be read.
    fn free_device_bytes(&self) -> Option<u64>;
    /// Finishes a save staged for the part that is about to leave.
    fn finish_save(&mut self) -> Result<()>;
    /// Runs before a session's decode: the host's per-job bookkeeping that its own `decode` does.
    fn before_decode(&mut self) {}
    /// Every id that ends generation: the host's shared handle.
    fn eos_ids(&self) -> Arc<[u32]>;
    /// The logits the sampler may choose from.
    fn sampler_vocab(&self) -> usize;
    /// The rows of the embedding table: a token at or above it is refused.
    fn embedding_rows(&self) -> usize;
    /// Whether the host steps several parts in one batched body ([`Self::step_parts_drawn`]).
    /// A host that does not keeps the windows time sliced, a row after a row.
    fn batched(&self) -> bool {
        false
    }
    /// Steps every part once with its token in one batched body, and draws the next token of
    /// each part from its own sampler on the device, in order. Called with two or more parts,
    /// each of which is out of the step (the step runs no part of its own). An error leaves
    /// every part in an unknown state.
    fn step_parts_drawn(
        &mut self,
        _parts: &mut [&mut Self::Part],
        _tokens: &[u32],
        _samplers: &[&Sampler],
    ) -> Result<Vec<u32>> {
        Err(EngError::Backend {
            operation: "pooled sessions",
            message: "this host has no batched step".to_owned(),
        })
    }
}

/// One session of a [`PooledSessions`] engine.
pub(crate) struct PooledSession<P> {
    core: DecodeSession,
    slot: Rc<SeqSlot<P>>,
    /// The number the session is named by in refusals.
    serial: u64,
}

/// The next session number: refusals name the session they refuse.
static NEXT_SERIAL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

fn next_serial() -> u64 {
    NEXT_SERIAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

/// The executor of several sessions over one resident step.
pub(crate) struct PooledSessions<E: PartHost> {
    // Declaration order is drop order: the parts (and their graphs) go before the step.
    /// The small part the step runs while no session does. It is `Some` only while a session's
    /// part runs in the step, and then it waits here.
    idle: Option<E::Part>,
    /// The session whose part the step runs now; `None` when the idle part runs.
    resident: Option<Rc<SeqSlot<E::Part>>>,
    pool: Arc<CellPool>,
    /// The most cells one session holds: the context the card plan chose.
    ceiling: usize,
    max_seqs: usize,
    /// Whether the step cannot host more than one session (tiered experts and the like keep
    /// state outside a session's part) and a start that named no count serves its one session
    /// over the step's own planes instead of refusing. No part is ever built or swapped.
    single: bool,
    /// Whether the last session request found the pool empty and said so.
    waiting: std::cell::Cell<bool>,
    /// The ids that end generation, and the sampler's vocabulary: the window's limits.
    eos: Arc<[u32]>,
    sampler_vocab: usize,
    step: E,
}

impl<E: PartHost> PooledSessions<E> {
    /// Takes over a built step and the session the engine opened with. The step's planes were
    /// sized for the whole context: they go back to the card and their cells become the pool.
    /// Every session starts with the cells its request needs and grows as it decodes.
    pub(crate) fn new(
        mut step: E,
        core: DecodeSession,
        max_seqs: usize,
        allow_single: bool,
    ) -> Result<(Self, PooledSession<E::Part>)> {
        // The planner funded a count beside the banks: the executor never admits more.
        let max_seqs = step
            .funded_sessions()
            .map_or(max_seqs, |funded| max_seqs.min(funded.max(1)));
        // A start that named no session count and whose step cannot host sessions serves one
        // session; a named count above one is refused by the step, as it asked for more.
        let single = allow_single && step.check_sessions().is_err();
        if !single {
            step.check_sessions()?;
        }
        let ceiling = step.context_cells();
        let pool = CellPool::new(ceiling);
        if single {
            // The step's own planes are the session's; the lease still follows the tokens it
            // has, and grows with them.
            let lease = pool.lease_for(INITIAL_CELLS.min(ceiling), ceiling)?;
            let slot = Rc::new(SeqSlot {
                inner: RefCell::new(SlotInner { part: None, lease }),
            });
            eprintln!(
                "{}: sessions: this step serves one session ({} cells); a start that names \
                 max_seqs above one needs the plain resident step",
                E::LABEL,
                pool.total()
            );
            let this = Self {
                idle: None,
                resident: Some(Rc::clone(&slot)),
                pool,
                ceiling,
                max_seqs: 1,
                single: true,
                waiting: std::cell::Cell::new(false),
                eos: step.eos_ids(),
                sampler_vocab: step.sampler_vocab(),
                step,
            };
            return Ok((
                this,
                PooledSession {
                    core,
                    slot,
                    serial: next_serial(),
                },
            ));
        }
        let initial = INITIAL_CELLS.min(ceiling);
        let small = step.part_bytes_at(initial);
        let lease = pool.lease_for(initial, ceiling)?;
        let mut first = step.new_part(initial)?;
        step.swap_part(&mut first);
        drop(first);
        let part = step.new_part(initial)?;
        let slot = Rc::new(SeqSlot {
            inner: RefCell::new(SlotInner {
                part: Some(part),
                lease,
            }),
        });
        eprintln!(
            "{}: sessions: one pool of {} cells, up to {ceiling} for one session ({} each when \
             {max_seqs} are live); a session holds the cells of its tokens ({small} B of state \
             at start) and grows as it decodes; a request the pool cannot fund waits",
            E::LABEL,
            pool.total(),
            even_share(ceiling, max_seqs)
        );
        let this = Self {
            idle: None,
            resident: None,
            pool,
            ceiling,
            max_seqs,
            single: false,
            waiting: std::cell::Cell::new(false),
            eos: step.eos_ids(),
            sampler_vocab: step.sampler_vocab(),
            step,
        };
        Ok((
            this,
            PooledSession {
                core,
                slot,
                serial: next_serial(),
            },
        ))
    }

    /// Makes `slot`'s part the one the step runs.
    fn activate(&mut self, slot: &Rc<SeqSlot<E::Part>>) -> Result<()> {
        if self
            .resident
            .as_ref()
            .is_some_and(|resident| Rc::ptr_eq(resident, slot))
        {
            return Ok(());
        }
        // A save staged for the part that leaves finishes before the parts change places.
        self.step.finish_save()?;
        let Some(mut part) = slot.inner.borrow_mut().part.take() else {
            return Err(EngError::Backend {
                operation: "qwen36_cuda_sessions",
                message: "the session's sequence part is missing".to_owned(),
            });
        };
        self.step.swap_part(&mut part);
        match self.resident.replace(Rc::clone(slot)) {
            Some(left) => left.inner.borrow_mut().part = Some(part),
            None => self.idle = Some(part),
        }
        Ok(())
    }

    /// Whether the session whose part runs in the step was dropped: the executor's own handle is
    /// then the only one left, and the part and the cells it holds belong to nobody.
    fn orphaned(&self) -> bool {
        self.resident
            .as_ref()
            .is_some_and(|resident| Rc::strong_count(resident) == 1)
    }

    /// Gives the planes and the cells of a dropped resident session back to the card and the
    /// pool. Every way a session can be dropped (a failed setup, a failed window, a pool that
    /// trims idle sessions) leaves it here, so one call at each entry covers them all.
    fn reclaim(&mut self) {
        if self.orphaned() {
            // A save the dropped session staged that cannot finish is lost with it; its planes
            // and cells go back all the same.
            if self.deactivate().is_err() {
                self.force_deactivate();
            }
        }
    }

    /// Puts the idle part under the step, so that the resident session's part is in its slot. A
    /// save staged for the part that leaves finishes first, as when a session takes the step:
    /// the staged copy names the part's lineage, which must still be in the step to finish.
    fn deactivate(&mut self) -> Result<()> {
        if self.resident.is_none() {
            return Ok(());
        }
        self.step.finish_save()?;
        self.force_deactivate();
        Ok(())
    }

    /// The exchange of [`Self::deactivate`], whatever became of a staged save.
    fn force_deactivate(&mut self) {
        let Some(left) = self.resident.take() else {
            return;
        };
        if self.single {
            // The step's own planes are the session's: nothing to swap back.
            return;
        }
        let mut idle = self.idle.take().expect("the idle part is present");
        self.step.swap_part(&mut idle);
        left.inner.borrow_mut().part = Some(idle);
    }

    /// Sizes the session's planes to `cells`: the old planes go to the card before the new ones
    /// are allocated, and the new planes start zeroed with no captured graph.
    fn resize(&mut self, session: &mut PooledSession<E::Part>, cells: usize) -> Result<()> {
        if self.single {
            // The planes are the step's own: only the cells the session is leased follow the
            // request.
            return session
                .slot
                .inner
                .borrow_mut()
                .lease
                .resize_for(cells, self.ceiling);
        }
        let resident = self
            .resident
            .as_ref()
            .is_some_and(|resident| Rc::ptr_eq(resident, &session.slot));
        let held = if resident {
            Some(self.step.context_cells())
        } else {
            session.slot.inner.borrow().part.as_ref().map(E::part_cells)
        };
        if session.slot.inner.borrow().lease.cells() == cells && held == Some(cells) {
            return Ok(());
        }
        self.deactivate()?;
        let small = INITIAL_CELLS.min(self.ceiling);
        session
            .slot
            .inner
            .borrow_mut()
            .lease
            .resize_for(cells, self.ceiling)?;
        drop(session.slot.inner.borrow_mut().part.take());
        match self.step.new_part(cells) {
            Ok(part) => {
                session.slot.inner.borrow_mut().part = Some(part);
                Ok(())
            }
            Err(error) => {
                // The planes could not be built: the session keeps the smallest planes it can,
                // and its lease follows them.
                let mut inner = session.slot.inner.borrow_mut();
                inner.lease.resize(small);
                inner.part = self.step.new_part(small).ok();
                Err(error)
            }
        }
    }

    /// Gives the session room for `positions` positions: the cells it holds are leased from the
    /// pool by half again what it needs (so a long decode grows a few times, not at every
    /// token) and its planes are re-laid in larger ones with the sequence intact. A session the
    /// pool cannot fund is refused alone: it keeps its planes and its cells, and no other
    /// session's cells are touched.
    fn make_room(&mut self, session: &PooledSession<E::Part>, positions: usize) -> Result<()> {
        let held = self.held_cells(session);
        if positions <= held {
            return Ok(());
        }
        let who = session.serial;
        let cells = fund_growth(
            &mut session.slot.inner.borrow_mut().lease,
            who,
            held,
            positions,
            self.ceiling,
        )?;
        if self.single {
            // The planes are the step's own and already hold the context: only the lease grew.
            return Ok(());
        }
        // The old planes stay until the copy ends: the card must hold both.
        let cells = fit_planes(
            &mut session.slot.inner.borrow_mut().lease,
            who,
            held,
            cells,
            positions,
            self.step.free_device_bytes(),
            |cells| self.step.part_bytes_at(cells),
        )?;
        if let Err(error) = self.deactivate_if_resident(session) {
            session.slot.inner.borrow_mut().lease.resize(held);
            return Err(error);
        }
        let began = std::time::Instant::now();
        let grown = {
            let mut inner = session.slot.inner.borrow_mut();
            match inner.part.as_mut() {
                Some(part) => self.step.grow_part(part, cells),
                None => Err(EngError::Backend {
                    operation: "pooled sessions",
                    message: "the session's sequence part is missing".to_owned(),
                }),
            }
        };
        match grown {
            Ok(()) => {
                eprintln!(
                    "{}: session {who} grew from {held} to {cells} cells in {:.1} ms ({} of {} cells free)",
                    E::LABEL,
                    began.elapsed().as_secs_f64() * 1000.0,
                    self.pool.free(),
                    self.pool.total()
                );
                Ok(())
            }
            // The planes are as they were: the lease follows them.
            Err(error) => {
                session.slot.inner.borrow_mut().lease.resize(held);
                Err(match error {
                    EngError::MemoryPlan(detail) => EngError::MemoryPlan(format!(
                        "session {who} cannot grow from {held} to {cells} cells on the card: \
                         {detail}; the request ends here and the others go on"
                    )),
                    other => other,
                })
            }
        }
    }

    /// The cells the session's planes hold now.
    fn held_cells(&self, session: &PooledSession<E::Part>) -> usize {
        if self.single {
            return session.slot.inner.borrow().lease.cells();
        }
        if self
            .resident
            .as_ref()
            .is_some_and(|resident| Rc::ptr_eq(resident, &session.slot))
        {
            self.step.context_cells()
        } else {
            session
                .slot
                .inner
                .borrow()
                .part
                .as_ref()
                .map_or(0, E::part_cells)
        }
    }

    /// Puts the session's part in its slot when the step runs it.
    fn deactivate_if_resident(&mut self, session: &PooledSession<E::Part>) -> Result<()> {
        if self
            .resident
            .as_ref()
            .is_some_and(|resident| Rc::ptr_eq(resident, &session.slot))
        {
            self.deactivate()?;
        }
        Ok(())
    }

    /// Device bytes the planes of one more session need at the start.
    fn free_device_bytes(&self) -> Option<u64> {
        self.step.free_device_bytes()
    }
}

impl<E: PartHost> ExecutorOps for PooledSessions<E> {
    type Session = PooledSession<E::Part>;

    fn max_seqs(&self) -> usize {
        self.max_seqs
    }

    /// The pool must hold a first lease, and the device must hold the planes with room to spare.
    fn can_admit_session(&self) -> bool {
        if self.single {
            // The one session's planes are free when nobody holds them (a failed admission
            // cleared them) or the holder was dropped.
            return self.resident.is_none() || self.orphaned();
        }
        // A dropped session's cells come back at the next admission.
        if self.pool.free() < INITIAL_CELLS && !self.orphaned() {
            return false;
        }
        let per_session = self.step.part_bytes_at(INITIAL_CELLS);
        self.free_device_bytes()
            .is_none_or(|free| free >= per_session.saturating_mul(2))
    }

    /// Prefix entries live in host memory and the sessions on the device: dropping an entry does
    /// not make room for a session.
    fn eviction_frees_session_memory(&self) -> bool {
        false
    }

    fn batch_policy(&self) -> BatchPolicy {
        BatchPolicy::default()
    }

    fn new_session(&mut self) -> Result<PooledSession<E::Part>> {
        self.reclaim();
        if self.single {
            // The step runs one session: a dropped one hands its planes to the next.
            if self.resident.is_some() {
                return Err(EngError::MemoryPlan(format!(
                    "{}: this step serves one session and it is busy",
                    E::LABEL
                )));
            }
            let lease = self
                .pool
                .lease_for(INITIAL_CELLS.min(self.ceiling), self.ceiling)?;
            let slot = Rc::new(SeqSlot {
                inner: RefCell::new(SlotInner { part: None, lease }),
            });
            self.resident = Some(Rc::clone(&slot));
            return Ok(PooledSession {
                core: DecodeSession::default(),
                slot,
                serial: next_serial(),
            });
        }
        let cells = INITIAL_CELLS.min(self.ceiling);
        let lease = match self.pool.lease_for(cells, self.ceiling) {
            Ok(lease) => lease,
            Err(error) => {
                if !self.waiting.replace(true) {
                    eprintln!(
                        "{}: a request waits for a session: the pool holds {} of {} cells \
                         free and the live sessions hold the rest; it runs when one finishes",
                        E::LABEL,
                        self.pool.free(),
                        self.pool.total()
                    );
                }
                return Err(error);
            }
        };
        self.waiting.set(false);
        let part = self.step.new_part(cells)?;
        Ok(PooledSession {
            core: DecodeSession::default(),
            serial: next_serial(),
            slot: Rc::new(SeqSlot {
                inner: RefCell::new(SlotInner {
                    part: Some(part),
                    lease,
                }),
            }),
        })
    }

    fn check_prompt(&self, prompt_tokens: usize) -> Result<()> {
        crate::family::session::check_prompt(prompt_tokens, self.ceiling)
    }

    fn prefix_limits(&self) -> StoreLimits {
        ExecutorOps::prefix_limits(&self.step)
    }

    fn history<'s>(&self, st: &'s PooledSession<E::Part>) -> &'s [u32] {
        st.core.history()
    }

    fn reset(&mut self, st: &mut PooledSession<E::Part>) -> Result<()> {
        self.activate(&st.slot)?;
        ExecutorOps::reset(&mut self.step, &mut st.core)
    }

    /// Sizes the session's cells to the request, leased from the pool: the tokens of the prompt
    /// and a little room, whatever generation limit the request names. The
    /// session grows from there as it decodes. A pool that cannot fund the prompt yet refuses
    /// with a memory-plan error and the worker queues the request until a session lets go. This
    /// runs before any prefix restore.
    fn reset_for_request(
        &mut self,
        st: &mut PooledSession<E::Part>,
        prompt_tokens: usize,
        completion_limit: Option<u32>,
        _has_prefix_boundary: bool,
    ) -> Result<()> {
        self.reclaim();
        let room = CellRoom {
            reachable: st.slot.inner.borrow().lease.reachable(),
            ceiling: self.ceiling,
            share: INITIAL_CELLS,
        };
        let request = CellRequest {
            prompt: prompt_tokens,
            completion: Some(admission_room(completion_limit)),
        };
        let cells = session_cells_or_queue(request, room)?;
        self.resize(st, cells)?;
        self.reset(st)
    }

    fn prefill(&mut self, st: &mut PooledSession<E::Part>, toks: &[u32]) -> Result<()> {
        // A prompt that leaves no room to generate is refused with the structured error the
        // server answers with; only then does the session ask the pool for room: one more cell
        // than its tokens.
        let total = st.core.history().len() + toks.len();
        crate::family::session::check_prompt(total, self.ceiling)?;
        self.make_room(st, total + 1)?;
        self.activate(&st.slot)?;
        ExecutorOps::prefill(&mut self.step, &mut st.core, toks)
    }

    fn prefill_piece(&self) -> Option<usize> {
        ExecutorOps::prefill_piece(&self.step)
    }

    fn decode(
        &mut self,
        st: &mut PooledSession<E::Part>,
        generation: &mut GenerationState,
    ) -> Result<DecodeBatch> {
        // The pending token takes the position after the ones fed: one more cell than it has.
        self.make_room(st, st.core.history().len())?;
        self.activate(&st.slot)?;
        self.step.before_decode();
        // The session ends at the cells one session can reach, not at the cells it holds now:
        // it grows before the step that needs more.
        let limits = Limits {
            eos: &self.eos,
            sampler_vocab: self.sampler_vocab,
            context_limit: self.ceiling,
        };
        st.core
            .decode(&mut RowStep(&mut self.step), generation, &limits)
    }

    /// A host with a batched body steps the window's rows in one step through the shared window
    /// driver; any other host runs the rows one after another on the shared step, each through
    /// the single-row decode. Either way a row's tokens are the ones the same request gets
    /// alone. A failed row is its own error; the step's state of that row is poisoned and the
    /// worker resets it.
    fn decode_window(
        &mut self,
        sessions: &mut [&mut PooledSession<E::Part>],
        generations: &mut [&mut GenerationState],
    ) -> Result<Vec<Result<DecodeBatch>>> {
        if sessions.is_empty()
            || sessions.len() != generations.len()
            || sessions.len() > self.max_seqs
        {
            return Err(EngError::BadRequest(
                "a decode window needs equal nonempty slices within max_seqs".into(),
            ));
        }
        if self.step.batched() {
            return crate::family::window::decode_window(self, sessions, generations);
        }
        Ok(sessions
            .iter_mut()
            .zip(generations.iter_mut())
            .map(|(session, generation)| self.decode(session, generation))
            .collect())
    }

    fn after_emit(&mut self) -> Result<()> {
        ExecutorOps::after_emit(&mut self.step)
    }

    fn boundary(&mut self, st: &mut PooledSession<E::Part>) {
        ExecutorOps::boundary(&mut self.step, &mut st.core);
    }

    fn background_pending(&self) -> bool {
        ExecutorOps::background_pending(&self.step)
    }

    fn background_step(&mut self) {
        ExecutorOps::background_step(&mut self.step);
    }

    fn background_drain(&mut self) {
        ExecutorOps::background_drain(&mut self.step);
    }

    fn save_prefix(
        &mut self,
        st: &PooledSession<E::Part>,
        key: &[u32],
        kind: SaveKind,
    ) -> Result<bool> {
        // The entry is exported from the part the step runs, so the session's part must be it.
        // The store takes a cache of any size of the same form, so a session below the whole
        // context saves too.
        let resident = self
            .resident
            .as_ref()
            .is_some_and(|resident| Rc::ptr_eq(resident, &st.slot));
        if !resident {
            return Ok(false);
        }
        ExecutorOps::save_prefix(&mut self.step, &st.core, key, kind)
    }

    fn prefix_report(&self) -> Option<PrefixReport> {
        ExecutorOps::prefix_report(&self.step)
    }

    fn restore_prefix(&mut self, st: &mut PooledSession<E::Part>, prompt: &[u32]) -> Result<usize> {
        self.activate(&st.slot)?;
        ExecutorOps::restore_prefix(&mut self.step, &mut st.core, prompt)
    }
}

impl<E: PartHost> Drop for PooledSessions<E> {
    fn drop(&mut self) {
        // Every part leaves the step through its slot before the step's own drop synchronizes.
        let _ = self.deactivate();
        self.force_deactivate();
    }
}

/// The model side of one row of a window: the shared step, with the row's part running in it.
pub(crate) struct RowStep<'a, E: PartHost>(&'a mut E);

impl<E: PartHost> TokenStepper for RowStep<'_, E> {
    const OPERATION: &'static str = <E as TokenStepper>::OPERATION;

    fn feed(&mut self, token: u32) -> Result<()> {
        TokenStepper::feed(self.0, token)
    }

    fn feed_rows(&mut self, tokens: &[u32]) -> Result<()> {
        TokenStepper::feed_rows(self.0, tokens)
    }

    fn logits(&mut self, token: u32) -> Result<Vec<f32>> {
        TokenStepper::logits(self.0, token)
    }

    fn next_token(
        &mut self,
        token: u32,
        sampler: &mut Sampler,
        sampler_vocab: usize,
    ) -> Result<u32> {
        TokenStepper::next_token(self.0, token, sampler, sampler_vocab)
    }

    fn fed(&self) -> usize {
        TokenStepper::fed(self.0)
    }

    fn reset(&mut self) -> Result<()> {
        TokenStepper::reset(self.0)
    }
}

/// The window seam: the sessions of a pooled engine are the rows of a window, and a host that
/// supplies [`PartHost::step_parts_drawn`] steps them in one body.
impl<E: PartHost> WindowRows for PooledSessions<E> {
    type Stepper<'a>
        = RowStep<'a, E>
    where
        Self: 'a;
    type Permit = ();
    const OPERATION: &'static str = <E as TokenStepper>::OPERATION;

    /// The row's part runs in the step: the driver calls `activate_row` before it reads the
    /// model side.
    fn activate_row(&mut self, session: &mut PooledSession<E::Part>) -> Result<()> {
        self.make_room(session, session.core.history().len())?;
        self.activate(&session.slot)
    }

    fn split<'a>(
        &'a mut self,
        session: &'a mut PooledSession<E::Part>,
    ) -> (&'a mut DecodeSession, RowStep<'a, E>, Limits<'a>) {
        let Self {
            step,
            eos,
            sampler_vocab,
            ceiling,
            ..
        } = self;
        let context_limit = *ceiling;
        (
            &mut session.core,
            RowStep(step),
            Limits {
                eos,
                sampler_vocab: *sampler_vocab,
                context_limit,
            },
        )
    }

    /// The row's consumed positions against its cells, as the solo decode compares them: a row
    /// that has consumed all but the last cell still takes one more token.
    fn check_row(&self, session: &PooledSession<E::Part>, token: u32) -> Result<()> {
        let resident = self
            .resident
            .as_ref()
            .is_some_and(|resident| Rc::ptr_eq(resident, &session.slot));
        let (cells, fed) = if resident {
            (self.step.context_cells(), TokenStepper::fed(&self.step))
        } else {
            session
                .slot
                .inner
                .borrow()
                .part
                .as_ref()
                .map_or((0, 0), |part| (E::part_cells(part), E::part_fed(part)))
        };
        if fed >= cells {
            return Err(EngError::ContextLimit(format!(
                "position {fed} reaches the {cells} cell context of the session"
            )));
        }
        if token as usize >= self.step.embedding_rows() {
            return Err(EngError::BadRequest(
                "token is outside the embedding table".to_owned(),
            ));
        }
        Ok(())
    }

    /// A row's planes grow in `activate_row`, before its step: the position itself needs nothing
    /// beyond them.
    fn row_growth(&self, _session: &PooledSession<E::Part>) -> Result<u64> {
        Ok(0)
    }

    fn reserve_growth(&self, _bytes: u64) -> Result<()> {
        Ok(())
    }

    /// The logits rows of a window with a request that records its draw: only the host walk has
    /// the candidates. The rows run one after another, each through the host's single-row logits.
    fn step_rows(
        &mut self,
        sessions: &mut [&mut PooledSession<E::Part>],
        tokens: &[u32],
    ) -> Result<Vec<Vec<f32>>> {
        let mut rows = Vec::with_capacity(sessions.len());
        for (session, &token) in sessions.iter_mut().zip(tokens) {
            self.activate(&session.slot)?;
            rows.push(TokenStepper::logits(&mut self.step, token)?);
        }
        Ok(rows)
    }

    /// Whether the host's batched body draws the window's tokens; a window with a recording
    /// request takes `step_rows` instead.
    fn draws_on_device(&self) -> bool {
        self.step.batched()
    }

    fn step_rows_drawn(
        &mut self,
        sessions: &mut [&mut PooledSession<E::Part>],
        tokens: &[u32],
        samplers: &[&Sampler],
    ) -> Result<Vec<u32>> {
        if tokens.len() != sessions.len() || samplers.len() != sessions.len() {
            return Err(EngError::BadRequest(
                "a window needs one token and one sampler per session".to_owned(),
            ));
        }
        // Every part leaves the step, so that the batched body reads them all from their slots.
        self.deactivate()?;
        let mut held: Vec<_> = sessions
            .iter()
            .map(|session| session.slot.inner.borrow_mut())
            .collect();
        let mut parts = Vec::with_capacity(held.len());
        for inner in &mut held {
            parts.push(inner.part.as_mut().ok_or_else(|| EngError::Backend {
                operation: "pooled sessions",
                message: "the session's sequence part is missing".to_owned(),
            })?);
        }
        self.step.step_parts_drawn(&mut parts, tokens, samplers)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::window::decode_window;
    use crate::tokenizer::{DecodeFinish, IncrementalDecoder, Tokenizer};

    const VOCAB: usize = 16;
    const EOS: u32 = 9;

    fn tokenizer() -> Tokenizer {
        let vocab: Vec<String> = (0..VOCAB)
            .map(|id| format!("\"{}\":{id}", char::from(b'a' + id as u8)))
            .collect();
        let json = format!(
            r#"{{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}},"post_processor":{{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}},"decoder":{{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}},"model":{{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":{{{}}},"merges":[]}}}}"#,
            vocab.join(",")
        );
        Tokenizer::from_bytes(json.as_bytes()).expect("tokenizer")
    }

    fn generation(limit: u32) -> GenerationState {
        GenerationState::new(IncrementalDecoder::new(&tokenizer())).with_max_tokens(limit)
    }

    /// The next token of a row: a function of its pending token alone.
    fn next_of(token: u32) -> u32 {
        (token + 1) % VOCAB as u32
    }

    /// A sequence part with no device: its cells and the positions it has consumed.
    #[derive(Clone, Copy, Debug)]
    struct FakePart {
        cells: usize,
        fed: usize,
    }

    /// A host with no device: the step runs one part, and a batched step draws `next_of`.
    struct Fake {
        /// Whether the step refuses to host sessions, as a tiered-experts step does.
        refuses: bool,
        running: FakePart,
        /// The cells of the parts of each batched step.
        batches: Vec<Vec<usize>>,
        /// Fails the finish of a save on this call of it, counting from one.
        fail_save_on: Option<usize>,
        saves: usize,
        /// Makes the growth of a part fail, as a card with no room does.
        fail_grow: bool,
        /// A save is staged and not finished.
        staged: bool,
        /// The cells of the part each save and each restore ran over.
        saved: Vec<usize>,
        restored: Vec<usize>,
        /// The free bytes of the fake card, when it can be read.
        free_bytes: Option<u64>,
    }

    impl Fake {
        fn new(refuses: bool) -> Self {
            Self {
                refuses,
                running: FakePart {
                    cells: POOL,
                    fed: 0,
                },
                batches: Vec::new(),
                fail_save_on: None,
                saves: 0,
                fail_grow: false,
                staged: false,
                saved: Vec::new(),
                restored: Vec::new(),
                free_bytes: None,
            }
        }
    }

    impl TokenStepper for Fake {
        const OPERATION: &'static str = "fake";

        fn feed(&mut self, _token: u32) -> Result<()> {
            self.running.fed += 1;
            Ok(())
        }

        fn logits(&mut self, token: u32) -> Result<Vec<f32>> {
            self.running.fed += 1;
            let mut row = vec![0.0; VOCAB];
            row[next_of(token) as usize] = 4.0;
            Ok(row)
        }

        fn fed(&self) -> usize {
            self.running.fed
        }

        fn reset(&mut self) -> Result<()> {
            self.running.fed = 0;
            Ok(())
        }
    }

    impl ExecutorOps for Fake {
        type Session = DecodeSession;

        fn history<'s>(&self, st: &'s DecodeSession) -> &'s [u32] {
            st.history()
        }

        fn reset(&mut self, st: &mut DecodeSession) -> Result<()> {
            st.reset(self)
        }

        fn prefill(&mut self, st: &mut DecodeSession, toks: &[u32]) -> Result<()> {
            let cells = self.running.cells;
            st.prefill(self, toks, cells)
        }

        fn decode(
            &mut self,
            st: &mut DecodeSession,
            generation: &mut GenerationState,
        ) -> Result<DecodeBatch> {
            let limits = Limits {
                eos: &[EOS],
                sampler_vocab: VOCAB,
                context_limit: self.running.cells,
            };
            st.decode(self, generation, &limits)
        }

        fn boundary(&mut self, _st: &mut DecodeSession) {}

        fn save_prefix(
            &mut self,
            _st: &DecodeSession,
            _key: &[u32],
            _kind: crate::family::session::SaveKind,
        ) -> Result<bool> {
            self.staged = true;
            self.saved.push(self.running.cells);
            Ok(true)
        }

        fn restore_prefix(&mut self, _st: &mut DecodeSession, prompt: &[u32]) -> Result<usize> {
            self.restored.push(self.running.cells);
            Ok(prompt.len() - 1)
        }
    }

    impl PartHost for Fake {
        type Part = FakePart;
        const LABEL: &'static str = "fake";

        fn check_sessions(&self) -> Result<()> {
            if self.refuses {
                return Err(EngError::BadRequest(
                    "the fake step refuses sessions".into(),
                ));
            }
            Ok(())
        }

        fn context_cells(&self) -> usize {
            self.running.cells
        }

        fn new_part(&mut self, cells: usize) -> Result<FakePart> {
            Ok(FakePart { cells, fed: 0 })
        }

        fn swap_part(&mut self, part: &mut FakePart) {
            std::mem::swap(&mut self.running, part);
        }

        fn grow_part(&mut self, part: &mut FakePart, cells: usize) -> Result<()> {
            if self.fail_grow {
                return Err(EngError::MemoryPlan("the card has no room".to_owned()));
            }
            part.cells = cells;
            Ok(())
        }

        fn part_cells(part: &FakePart) -> usize {
            part.cells
        }

        fn part_fed(part: &FakePart) -> usize {
            part.fed
        }

        fn part_bytes_at(&self, cells: usize) -> u64 {
            cells as u64
        }

        fn free_device_bytes(&self) -> Option<u64> {
            self.free_bytes
        }

        fn finish_save(&mut self) -> Result<()> {
            self.saves += 1;
            self.staged = false;
            if self.fail_save_on == Some(self.saves) {
                return Err(EngError::Backend {
                    operation: "fake",
                    message: "the staged save failed".to_owned(),
                });
            }
            Ok(())
        }

        fn eos_ids(&self) -> Arc<[u32]> {
            Arc::from(vec![EOS])
        }

        fn sampler_vocab(&self) -> usize {
            VOCAB
        }

        fn embedding_rows(&self) -> usize {
            VOCAB
        }

        fn batched(&self) -> bool {
            true
        }

        fn step_parts_drawn(
            &mut self,
            parts: &mut [&mut FakePart],
            tokens: &[u32],
            samplers: &[&Sampler],
        ) -> Result<Vec<u32>> {
            assert_eq!(samplers.len(), parts.len());
            self.batches
                .push(parts.iter().map(|part| part.cells).collect());
            for part in parts.iter_mut() {
                part.fed += 1;
            }
            Ok(tokens.iter().map(|&token| next_of(token)).collect())
        }
    }

    const POOL: usize = 110 * 1024;

    type Engine = PooledSessions<Fake>;
    type Session = PooledSession<FakePart>;

    fn engine(max_seqs: usize) -> (Engine, Session) {
        PooledSessions::new(Fake::new(false), DecodeSession::default(), max_seqs, false)
            .expect("sessions")
    }

    fn refusing(max_seqs: usize, allow_single: bool) -> Result<(Engine, Session)> {
        PooledSessions::new(
            Fake::new(true),
            DecodeSession::default(),
            max_seqs,
            allow_single,
        )
    }

    /// A session of the engine that has consumed `prompt`, leaving its last token pending.
    fn prefilled(engine: &mut Engine, mut session: Session, prompt: &[u32]) -> Session {
        engine
            .reset_for_request(&mut session, prompt.len(), None, false)
            .expect("admitted");
        engine.prefill(&mut session, prompt).expect("prefill");
        session
    }

    fn next_session(engine: &mut Engine, prompt: &[u32]) -> Session {
        let session = engine.new_session().expect("a session");
        prefilled(engine, session, prompt)
    }

    /// Runs one window of the live sessions and returns each row's result.
    fn window(
        engine: &mut Engine,
        sessions: &mut [&mut Session],
        generations: &mut [&mut GenerationState],
    ) -> Vec<Result<DecodeBatch>> {
        decode_window(engine, sessions, generations).expect("window")
    }

    /// The tokens one prompt generates alone on a one-session engine.
    fn solo(prompt: &[u32], limit: u32) -> Vec<u32> {
        let (mut engine, first) = engine(1);
        let mut session = prefilled(&mut engine, first, prompt);
        let mut generation = generation(limit);
        let mut tokens = Vec::new();
        loop {
            let batch = engine
                .decode(&mut session, &mut generation)
                .expect("decode");
            tokens.extend(&batch.tokens);
            if batch.finish.is_some() {
                return tokens;
            }
        }
    }

    #[test]
    fn a_session_dropped_after_a_failed_setup_gives_its_cells_back() {
        let (mut sessions, mut first) = engine(8);
        sessions
            .reset_for_request(&mut first, 100_000, None, false)
            .expect("the first request holds its prompt");
        assert!(sessions.pool.free() < POOL / 8);
        // The worker drops a session whose setup failed; its part still runs in the step.
        drop(first);
        assert!(
            sessions.can_admit_session(),
            "the dropped session's cells are reclaimable"
        );
        let mut second = sessions
            .new_session()
            .expect("the next request is admitted");
        assert_eq!(sessions.pool.free(), POOL - INITIAL_CELLS);
        sessions
            .reset_for_request(&mut second, 100_000, None, false)
            .expect("and holds the same prompt");
    }

    #[test]
    fn a_dropped_session_does_not_refuse_a_long_prompt_on_an_empty_pool() {
        let (mut sessions, mut first) = engine(2);
        sessions
            .reset_for_request(&mut first, 100, None, false)
            .expect("the first request holds its prompt and a little room");
        assert_eq!(sessions.step.context_cells(), 2 * INITIAL_CELLS);
        drop(first);
        let mut lone = sessions.new_session().expect("admitted");
        sessions
            .reset_for_request(&mut lone, 70_000, None, false)
            .expect("a long prompt is served while the pool is empty");
        assert!(sessions.step.context_cells() >= 70_000);
    }

    #[test]
    fn a_live_session_keeps_its_cells() {
        let (mut sessions, mut first) = engine(8);
        sessions
            .reset_for_request(&mut first, 112_000, None, false)
            .expect("the first request holds the pool");
        assert!(!sessions.orphaned());
        assert_eq!(sessions.pool.free(), 0);
        assert!(matches!(
            sessions.new_session(),
            Err(EngError::MemoryPlan(_))
        ));
    }

    #[test]
    fn a_session_grows_with_its_prompt_and_keeps_where_it_was() {
        let (mut sessions, _first) = engine(4);
        let mut grower = sessions.new_session().expect("a session");
        assert_eq!(sessions.held_cells(&grower), INITIAL_CELLS);
        let prompt = vec![1u32; 3000];
        sessions
            .prefill(&mut grower, &prompt[..1000])
            .expect("fits");
        assert_eq!(sessions.held_cells(&grower), INITIAL_CELLS);
        sessions
            .prefill(&mut grower, &prompt[1000..])
            .expect("grows");
        let held = sessions.held_cells(&grower);
        assert!(held > 3000 && held % INITIAL_CELLS == 0, "{held}");
        assert_eq!(sessions.pool.free(), POOL - INITIAL_CELLS - held);
        // The sequence went on where it was: it consumed all but the pending token.
        sessions.deactivate().expect("the part goes to its slot");
        let part = grower.slot.inner.borrow().part.expect("part");
        assert_eq!(part.fed, 2999);
    }

    #[test]
    fn a_window_of_rows_equals_each_row_alone_through_one_batched_step() {
        let (mut engine, first) = engine(4);
        let mut a = prefilled(&mut engine, first, &[1, 2]);
        let mut b = next_session(&mut engine, &[5]);
        let mut c = next_session(&mut engine, &[0, 3, 4]);
        let (mut ga, mut gb, mut gc) = (generation(12), generation(12), generation(12));
        let mut got = [Vec::new(), Vec::new(), Vec::new()];
        let mut live = vec![0usize, 1, 2];
        while !live.is_empty() {
            let batches = {
                let mut rows: Vec<&mut Session> = Vec::new();
                let mut gens: Vec<&mut GenerationState> = Vec::new();
                for (index, (session, generation)) in
                    [(&mut a, &mut ga), (&mut b, &mut gb), (&mut c, &mut gc)]
                        .into_iter()
                        .enumerate()
                {
                    if live.contains(&index) {
                        rows.push(session);
                        gens.push(generation);
                    }
                }
                window(&mut engine, &mut rows, &mut gens)
            };
            let mut finished = Vec::new();
            for (&index, batch) in live.iter().zip(batches) {
                let batch = batch.expect("row");
                got[index].extend(&batch.tokens);
                if batch.finish.is_some() {
                    finished.push(index);
                }
            }
            live.retain(|index| !finished.contains(index));
        }
        assert_eq!(got[0], solo(&[1, 2], 12));
        assert_eq!(got[1], solo(&[5], 12));
        assert_eq!(got[2], solo(&[0, 3, 4], 12));
        // The prompt [5] runs into EOS (9) mid-window: 6, 7, 8, 9.
        assert_eq!(got[1].last(), Some(&EOS));
        assert!(!engine.step.batches.is_empty(), "the rows were batched");
    }

    #[test]
    fn a_row_that_reaches_its_limit_ends_alone() {
        let (mut engine, first) = engine(2);
        let mut a = prefilled(&mut engine, first, &[1]);
        let mut b = next_session(&mut engine, &[2]);
        let (mut ga, mut gb) = (generation(2), generation(8));
        let first_window = window(&mut engine, &mut [&mut a, &mut b], &mut [&mut ga, &mut gb]);
        assert!(
            first_window
                .iter()
                .all(|row| row.as_ref().is_ok_and(|b| b.finish.is_none()))
        );
        let second = window(&mut engine, &mut [&mut a, &mut b], &mut [&mut ga, &mut gb]);
        let ended = second[0].as_ref().expect("row a");
        assert!(
            matches!(ended.finish, Some(DecodeFinish::Length)),
            "{ended:?}"
        );
        let alive = second[1].as_ref().expect("row b");
        assert!(alive.finish.is_none());
        assert!(!alive.tokens.is_empty());
    }

    #[test]
    fn a_row_with_a_recording_request_takes_the_logits_rows_and_the_others_still_serve() {
        let (mut engine, first) = engine(2);
        let mut a = prefilled(&mut engine, first, &[1, 2]);
        let mut b = next_session(&mut engine, &[5]);
        let (mut ga, mut gb) = (generation(4), generation(4));
        let _capture = gb.sampler.arm_capture(8);
        let results = window(&mut engine, &mut [&mut a, &mut b], &mut [&mut ga, &mut gb]);
        let (ra, rb) = (
            results[0].as_ref().expect("a"),
            results[1].as_ref().expect("b"),
        );
        assert_eq!(ra.tokens, vec![next_of(2)]);
        assert_eq!(rb.tokens, vec![next_of(5)]);
        assert!(
            engine.step.batches.is_empty(),
            "the window took the host walk"
        );
        // Once the recording request is gone, the same sessions batch again.
        let (mut gc, mut gd) = (generation(4), generation(4));
        let again = window(&mut engine, &mut [&mut a, &mut b], &mut [&mut gc, &mut gd]);
        assert_eq!(again.len(), 2);
        assert_eq!(engine.step.batches.len(), 1);
    }

    #[test]
    fn a_row_that_cannot_be_brought_to_the_step_is_refused_alone() {
        let (mut engine, first) = engine(3);
        let mut a = prefilled(&mut engine, first, &[1, 2]);
        let mut b = next_session(&mut engine, &[5]);
        let mut c = next_session(&mut engine, &[0, 3]);
        let (mut ga, mut gb, mut gc) = (generation(4), generation(4), generation(4));
        // The finish of the staged save fails when the window first brings the second row.
        engine.step.fail_save_on = Some(engine.step.saves + 2);
        let results = window(
            &mut engine,
            &mut [&mut a, &mut b, &mut c],
            &mut [&mut ga, &mut gb, &mut gc],
        );
        assert!(results[0].is_ok(), "row a serves");
        assert!(results[1].is_err(), "row b is refused alone");
        assert!(results[2].is_ok(), "row c serves");
        // The refusal belongs to that window: the next one serves every row.
        let next = window(
            &mut engine,
            &mut [&mut a, &mut b, &mut c],
            &mut [&mut ga, &mut gb, &mut gc],
        );
        assert!(next.iter().all(Result::is_ok));
    }

    #[test]
    fn a_row_that_fills_its_cells_grows_and_goes_on() {
        let (mut engine, first) = engine(2);
        let prompt = vec![1u32; INITIAL_CELLS - 1];
        let mut a = engine.new_session().expect("a");
        engine.prefill(&mut a, &prompt).expect("prefill a");
        let mut b = first;
        engine.prefill(&mut b, &prompt).expect("prefill b");
        let (mut ga, mut gb) = (generation(8), generation(8));
        for _ in 0..3 {
            let step = window(&mut engine, &mut [&mut a, &mut b], &mut [&mut ga, &mut gb]);
            assert!(step.iter().all(Result::is_ok), "{step:?}");
        }
        // Past 1024 positions each row holds more cells, and its tokens are those of a lone run.
        assert!(engine.held_cells(&a) > INITIAL_CELLS);
        assert!(engine.held_cells(&b) > INITIAL_CELLS);
    }

    #[test]
    fn a_row_the_pool_cannot_grow_is_refused_alone_and_no_neighbour_is_touched() {
        let (mut engine, first) = engine(3);
        let prompt = vec![1u32; INITIAL_CELLS - 1];
        let mut small = first;
        engine.prefill(&mut small, &prompt).expect("prefill");
        let mut big = engine.new_session().expect("big");
        engine
            .reset_for_request(&mut big, 111_000, None, false)
            .expect("the second request takes the rest of the pool");
        assert_eq!(engine.pool.free(), 0);
        let big_prompt = vec![2u32; 111_000];
        engine.prefill(&mut big, &big_prompt).expect("prefill big");
        let (mut gs, mut gb) = (generation(8), generation(8));
        let before = engine.held_cells(&big);
        let mut refused = false;
        for _ in 0..4 {
            let step = window(
                &mut engine,
                &mut [&mut small, &mut big],
                &mut [&mut gs, &mut gb],
            );
            assert!(step[1].is_ok(), "the neighbour goes on: {step:?}");
            if let Err(error) = &step[0] {
                assert!(matches!(error, EngError::MemoryPlan(_)), "{error:?}");
                refused = true;
                break;
            }
        }
        assert!(refused, "the row that cannot grow is refused");
        assert_eq!(
            engine.held_cells(&small),
            INITIAL_CELLS,
            "it keeps its planes"
        );
        assert_eq!(
            engine.held_cells(&big),
            before,
            "the neighbour's cells are untouched"
        );
    }

    #[test]
    fn a_growth_that_fails_on_the_card_leaves_the_session_as_it_was() {
        let (mut engine, first) = engine(2);
        let mut a = first;
        engine
            .prefill(&mut a, &vec![1u32; INITIAL_CELLS - 1])
            .expect("fits");
        engine.step.fail_grow = true;
        let free = engine.pool.free();
        let result = engine.prefill(&mut a, &[1, 2, 3]);
        assert!(matches!(result, Err(EngError::MemoryPlan(_))), "{result:?}");
        assert_eq!(engine.held_cells(&a), INITIAL_CELLS);
        assert_eq!(engine.pool.free(), free, "the lease follows the planes");
    }

    /// A start that named no session count and whose step cannot host sessions (tiered experts)
    /// serves its one session over the step's own planes: the executor builds, serves one request
    /// with the whole pool, and says busy to a second one.
    #[test]
    fn a_step_that_cannot_host_sessions_serves_one_for_a_start_that_named_no_count() {
        let (mut sessions, mut first) = refusing(8, true).expect("one session");
        assert_eq!(sessions.max_seqs(), 1);
        sessions
            .reset_for_request(&mut first, 100, None, false)
            .expect("the one session takes the pool");
        assert_eq!(sessions.step.context_cells(), POOL);
        assert!(!sessions.can_admit_session());
        assert!(matches!(
            sessions.new_session(),
            Err(EngError::MemoryPlan(_))
        ));
    }

    /// The worker drops a session whose setup failed and asks for another: the planes pass on.
    #[test]
    fn a_dropped_one_session_hands_its_planes_to_the_next() {
        let (mut sessions, first) = refusing(8, true).expect("one session");
        drop(first);
        assert!(sessions.can_admit_session());
        let mut next = sessions.new_session().expect("the planes pass on");
        sessions
            .reset_for_request(&mut next, 100, None, false)
            .expect("and serve a request");
        assert_eq!(sessions.step.context_cells(), POOL);
        assert_eq!(sessions.pool.free(), POOL - 2 * INITIAL_CELLS);
    }

    /// The one session of a step that cannot host more holds the cells of its tokens too: its
    /// lease grows through the shared rule as it decodes, the step's own planes stay as they are,
    /// and a growth the pool cannot fund refuses that session alone with its cells unchanged.
    #[test]
    fn the_one_session_of_a_single_step_grows_its_cells_as_it_decodes() {
        let (mut sessions, mut one) = refusing(8, true).expect("one session");
        sessions
            .reset_for_request(&mut one, 100, None, false)
            .expect("sized to the prompt and a little room");
        assert_eq!(sessions.held_cells(&one), 2 * INITIAL_CELLS);
        assert_eq!(sessions.pool.free(), POOL - 2 * INITIAL_CELLS);
        sessions
            .prefill(&mut one, &vec![1u32; 3000])
            .expect("the prompt grows the lease");
        let held = sessions.held_cells(&one);
        assert!(held > 3000 && held % INITIAL_CELLS == 0, "{held}");
        assert_eq!(sessions.pool.free(), POOL - held);
        assert_eq!(
            sessions.step.context_cells(),
            POOL,
            "the planes are the step's own"
        );
        // A pool that cannot fund the next growth refuses it and leaves the lease as it was.
        let rest = sessions.pool.free();
        let hold = sessions
            .pool
            .lease_for(rest, sessions.ceiling)
            .expect("the rest of the pool");
        let result = sessions.prefill(&mut one, &vec![2u32; held]);
        assert!(matches!(result, Err(EngError::MemoryPlan(_))), "{result:?}");
        assert_eq!(sessions.held_cells(&one), held);
        drop(hold);
    }

    /// A failed `new_session` after the dropped session was reclaimed leaves the step with no
    /// resident session: the next request must still be admitted.
    #[test]
    fn a_failed_new_session_does_not_lock_the_one_session_step() {
        let (mut sessions, first) = refusing(8, true).expect("one session");
        drop(first);
        // The reclaim in `new_session` frees the planes; a pool that cannot lease fails the call.
        sessions.reclaim();
        assert!(sessions.resident.is_none());
        let hold = sessions
            .pool
            .lease_for(sessions.ceiling, sessions.ceiling)
            .expect("the pool is whole");
        assert!(matches!(
            sessions.new_session(),
            Err(EngError::MemoryPlan(_))
        ));
        assert!(
            sessions.can_admit_session(),
            "no resident session means the step is free"
        );
        drop(hold);
        let mut second = sessions
            .new_session()
            .expect("the second request is admitted");
        sessions
            .reset_for_request(&mut second, 100, None, false)
            .expect("and served");
    }

    /// A start that named more than one session asked for what the step cannot do: it is refused.
    #[test]
    fn a_named_session_count_on_a_step_that_cannot_host_sessions_is_refused() {
        assert!(matches!(refusing(4, false), Err(EngError::BadRequest(_))));
    }

    /// A session grown from a small start sits below the ceiling: the prefix store still takes its
    /// entry, and a second session restores it.
    #[test]
    fn a_session_below_the_ceiling_saves_and_a_second_restores_the_entry() {
        let (mut engine, mut a) = engine(4);
        engine
            .reset_for_request(&mut a, 100, None, false)
            .expect("a is sized to its prompt");
        engine.prefill(&mut a, &[1, 2, 3, 4]).expect("prefill");
        assert!(engine.held_cells(&a) < POOL);
        let saved = engine
            .save_prefix(&a, &[1, 2, 3, 4], crate::family::session::SaveKind::Tip)
            .expect("save");
        assert!(saved, "a session below the ceiling saves");
        assert_eq!(engine.step.saved, vec![2 * INITIAL_CELLS]);
        let mut b = engine.new_session().expect("b");
        engine
            .reset_for_request(&mut b, 100, None, false)
            .expect("b is sized to its prompt");
        let covered = engine
            .restore_prefix(&mut b, &[1, 2, 3, 4, 5])
            .expect("restore");
        assert_eq!(covered, 4);
        assert_eq!(engine.step.restored, vec![2 * INITIAL_CELLS]);
    }

    /// A growth between a staged save and its finish completes the save first: the staged copy
    /// names the lineage of the part that is leaving the step.
    #[test]
    fn a_session_grown_after_a_staged_save_completes_the_save_first() {
        let (mut engine, mut a) = engine(2);
        engine
            .reset_for_request(&mut a, 100, None, false)
            .expect("sized");
        engine.prefill(&mut a, &[1, 2, 3]).expect("prefill");
        assert!(
            engine
                .save_prefix(&a, &[1, 2, 3], crate::family::session::SaveKind::Tip)
                .expect("staged")
        );
        assert!(engine.step.staged);
        let before = engine.step.saves;
        engine
            .prefill(&mut a, &vec![1u32; 3000])
            .expect("grows past its cells");
        assert!(
            !engine.step.staged,
            "the save finished before the planes moved"
        );
        assert!(engine.step.saves > before);
    }

    /// A staged save that cannot finish refuses the growth, and the session keeps its cells.
    #[test]
    fn a_growth_whose_staged_save_cannot_finish_leaves_the_session_as_it_was() {
        let (mut engine, mut a) = engine(2);
        engine
            .reset_for_request(&mut a, 100, None, false)
            .expect("sized");
        engine.prefill(&mut a, &[1, 2, 3]).expect("prefill");
        assert!(
            engine
                .save_prefix(&a, &[1, 2, 3], crate::family::session::SaveKind::Tip)
                .expect("staged")
        );
        engine.step.fail_save_on = Some(engine.step.saves + 1);
        let held = engine.held_cells(&a);
        let free = engine.pool.free();
        assert!(engine.prefill(&mut a, &vec![1u32; 3000]).is_err());
        assert_eq!(engine.held_cells(&a), held);
        assert_eq!(engine.pool.free(), free, "the lease follows the planes");
    }

    /// The old planes stay until the copy ends, so a card with room for the new planes alone is
    /// too short: the session retries at exactly the cells it needs, and a card short of that
    /// refuses it, naming the session and the bytes.
    #[test]
    fn a_growth_the_card_cannot_hold_beside_the_old_planes_retries_at_the_need_and_then_refuses() {
        let (mut engine, mut a) = engine(2);
        let mut b = engine.new_session().expect("b");
        // Room for 1500 cells of planes but not for the 2048 the rule asks for.
        engine.step.free_bytes = Some(1600);
        engine
            .prefill(&mut b, &vec![1u32; 1500])
            .expect("grows to exactly its need");
        assert_eq!(engine.held_cells(&b), 1501);
        // Room for less than the need: refused, the session as it was.
        engine.step.free_bytes = Some(1200);
        let held = engine.held_cells(&a);
        let free = engine.pool.free();
        let result = engine.prefill(&mut a, &vec![1u32; 3000]);
        let Err(EngError::MemoryPlan(message)) = result else {
            panic!("a memory-plan refusal, got {result:?}");
        };
        assert!(message.contains("session"), "{message}");
        assert!(message.contains("1200 B"), "{message}");
        assert_eq!(engine.held_cells(&a), held);
        assert_eq!(engine.pool.free(), free);
    }

    /// An oversize prompt gets the structured error the server compacts to, not the pool's text.
    #[test]
    fn a_prompt_above_the_ceiling_is_refused_with_the_structured_error() {
        let (mut engine, mut a) = engine(2);
        let result = engine.prefill(&mut a, &vec![1u32; POOL]);
        assert!(
            matches!(
                result,
                Err(EngError::PromptTooLong {
                    limit_tokens,
                    ..
                }) if limit_tokens == (POOL - 1) as u64
            ),
            "{result:?}"
        );
    }

    /// A refusal to grow names the session and the shortfall.
    #[test]
    fn a_refusal_to_grow_names_the_session_and_the_cells_it_lacks() {
        let (mut engine, first) = engine(3);
        let mut small = first;
        engine
            .prefill(&mut small, &vec![1u32; INITIAL_CELLS - 1])
            .expect("fits");
        let mut big = engine.new_session().expect("big");
        engine
            .reset_for_request(&mut big, 111_000, None, false)
            .expect("takes the rest");
        let result = engine.prefill(&mut small, &vec![1u32; INITIAL_CELLS + 5]);
        let Err(EngError::MemoryPlan(message)) = result else {
            panic!("a memory-plan refusal, got {result:?}");
        };
        assert!(
            message.contains(&format!("session {}", small.serial)),
            "{message}"
        );
        assert!(message.contains("more cells"), "{message}");
    }
}
