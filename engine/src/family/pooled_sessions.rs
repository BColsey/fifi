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
            CellLease, CellPool, CellRequest, CellRoom, INITIAL_CELLS, even_share,
            session_cells_or_queue,
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
    /// The cells of the context the step was built for, which become the pool.
    fn context_cells(&self) -> usize;
    /// A zeroed part holding `cells` positions.
    fn new_part(&mut self, cells: usize) -> Result<Self::Part>;
    /// Exchanges the part the step runs with `part`.
    fn swap_part(&mut self, part: &mut Self::Part);
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

/// How a session's cells are sized until a session can grow.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Sizing {
    /// A session takes every cell it can reach: the first one the whole context it would get
    /// alone, later ones what is left. A request the pool cannot fund waits for a session to
    /// let go; no live session is shrunk.
    WholePool,
    /// The pool is split among `max_seqs` sessions by what each request needs.
    EvenShare,
}

/// One session of a [`PooledSessions`] engine.
pub(crate) struct PooledSession<P> {
    core: DecodeSession,
    slot: Rc<SeqSlot<P>>,
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
    /// The even share of the pool among the sessions.
    share: usize,
    max_seqs: usize,
    sizing: Sizing,
    /// Whether the last session request found the pool empty and said so.
    waiting: std::cell::Cell<bool>,
    /// The ids that end generation, and the sampler's vocabulary: the window's limits.
    eos: Arc<[u32]>,
    sampler_vocab: usize,
    step: E,
}

impl<E: PartHost> PooledSessions<E> {
    /// Takes over a built step and the session the engine opened with. The step's planes were
    /// sized for the whole context and become the pool's cells. With [`Sizing::WholePool`] the
    /// first session keeps those planes, so a lone client runs the state it always ran; with
    /// [`Sizing::EvenShare`] they go back to the card and every session leases by its request.
    pub(crate) fn new(
        mut step: E,
        core: DecodeSession,
        max_seqs: usize,
        sizing: Sizing,
    ) -> Result<(Self, PooledSession<E::Part>)> {
        step.check_sessions()?;
        let ceiling = step.context_cells();
        let pool = CellPool::new(ceiling);
        let initial = INITIAL_CELLS.min(ceiling);
        let share = even_share(ceiling, max_seqs);
        let small = step.part_bytes_at(initial);
        let (idle, resident, slot) = match sizing {
            Sizing::WholePool => {
                let lease = pool.lease_for(ceiling, ceiling)?;
                let idle = step.new_part(initial)?;
                let slot = Rc::new(SeqSlot {
                    inner: RefCell::new(SlotInner { part: None, lease }),
                });
                (Some(idle), Some(Rc::clone(&slot)), slot)
            }
            Sizing::EvenShare => {
                let lease = pool.lease_for(initial, ceiling)?;
                let mut small = step.new_part(initial)?;
                step.swap_part(&mut small);
                drop(small);
                let part = step.new_part(initial)?;
                let slot = Rc::new(SeqSlot {
                    inner: RefCell::new(SlotInner {
                        part: Some(part),
                        lease,
                    }),
                });
                (None, None, slot)
            }
        };
        match sizing {
            Sizing::WholePool => eprintln!(
                "{}: sessions: one pool of {} cells; the first request takes the whole \
                 context, later ones take what is left or wait for a session to finish \
                 (at most {max_seqs}; set max_seqs to split the pool evenly)",
                E::LABEL,
                pool.total()
            ),
            Sizing::EvenShare => eprintln!(
                "{}: sessions {max_seqs}: one pool of {} cells, up to {ceiling} for one \
                 session, even share {share}, {small} B of state per session at start",
                E::LABEL,
                pool.total()
            ),
        }
        let this = Self {
            idle,
            resident,
            pool,
            ceiling,
            share,
            max_seqs,
            sizing,
            waiting: std::cell::Cell::new(false),
            eos: step.eos_ids(),
            sampler_vocab: step.sampler_vocab(),
            step,
        };
        Ok((this, PooledSession { core, slot }))
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
            self.deactivate();
        }
    }

    /// Puts the idle part under the step, so that the resident session's part is in its slot.
    fn deactivate(&mut self) {
        let Some(left) = self.resident.take() else {
            return;
        };
        let mut idle = self.idle.take().expect("the idle part is present");
        self.step.swap_part(&mut idle);
        left.inner.borrow_mut().part = Some(idle);
    }

    /// Sizes the session's planes to `cells`: the old planes go to the card before the new ones
    /// are allocated, and the new planes start zeroed with no captured graph.
    fn resize(&mut self, session: &mut PooledSession<E::Part>, cells: usize) -> Result<()> {
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
        self.deactivate();
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

    /// Sizes the session's cells to the request, leased from the pool: a request that names its
    /// generation limit holds the prompt and the limit, one that names none holds the even share,
    /// and a prompt above the share holds what it needs while the pool can fund it. A pool that
    /// cannot fund the prompt yet refuses with a memory-plan error and the worker queues the
    /// request until a session lets go. This runs before any prefix restore.
    fn reset_for_request(
        &mut self,
        st: &mut PooledSession<E::Part>,
        prompt_tokens: usize,
        completion_limit: Option<u32>,
        _has_prefix_boundary: bool,
    ) -> Result<()> {
        self.reclaim();
        let reachable = st.slot.inner.borrow().lease.reachable();
        let room = CellRoom {
            reachable,
            ceiling: self.ceiling,
            share: match self.sizing {
                Sizing::WholePool => reachable,
                Sizing::EvenShare => self.share,
            },
        };
        // Until a session can grow, a whole-pool session holds every cell it can reach whatever
        // limit the request names, as a lone client always had.
        let completion = match self.sizing {
            Sizing::WholePool => None,
            Sizing::EvenShare => completion_limit,
        };
        let request = CellRequest {
            prompt: prompt_tokens,
            completion,
        };
        let cells = session_cells_or_queue(request, room)?;
        self.resize(st, cells)?;
        self.reset(st)
    }

    fn prefill(&mut self, st: &mut PooledSession<E::Part>, toks: &[u32]) -> Result<()> {
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
        self.activate(&st.slot)?;
        ExecutorOps::decode(&mut self.step, &mut st.core, generation)
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
        // The entry is exported from the part the step runs, so the session's part must be it,
        // and the prefix store is laid out for the whole context: a session cut to fewer cells
        // saves nothing.
        let resident = self
            .resident
            .as_ref()
            .is_some_and(|resident| Rc::ptr_eq(resident, &st.slot));
        if !resident || self.step.context_cells() != self.ceiling {
            return Ok(false);
        }
        ExecutorOps::save_prefix(&mut self.step, &st.core, key, kind)
    }

    fn prefix_report(&self) -> Option<PrefixReport> {
        ExecutorOps::prefix_report(&self.step)
    }

    fn restore_prefix(&mut self, st: &mut PooledSession<E::Part>, prompt: &[u32]) -> Result<usize> {
        self.activate(&st.slot)?;
        if self.step.context_cells() != self.ceiling {
            return Ok(0);
        }
        ExecutorOps::restore_prefix(&mut self.step, &mut st.core, prompt)
    }
}

impl<E: PartHost> Drop for PooledSessions<E> {
    fn drop(&mut self) {
        // Every part leaves the step through its slot before the step's own drop synchronizes.
        self.deactivate();
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
        self.activate(&session.slot)
    }

    fn split<'a>(
        &'a mut self,
        session: &'a mut PooledSession<E::Part>,
    ) -> (&'a mut DecodeSession, RowStep<'a, E>, Limits<'a>) {
        let context_limit = self.step.context_cells();
        let Self {
            step,
            eos,
            sampler_vocab,
            ..
        } = self;
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

    /// The planes are allocated at admission; a position costs no memory.
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
        self.step.finish_save()?;
        self.deactivate();
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
        running: FakePart,
        /// The cells of the parts of each batched step.
        batches: Vec<Vec<usize>>,
        /// Fails the finish of a save on this call of it, counting from one.
        fail_save_on: Option<usize>,
        saves: usize,
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
    }

    impl PartHost for Fake {
        type Part = FakePart;
        const LABEL: &'static str = "fake";

        fn check_sessions(&self) -> Result<()> {
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
            None
        }

        fn finish_save(&mut self) -> Result<()> {
            self.saves += 1;
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

    fn engine(max_seqs: usize, sizing: Sizing) -> (Engine, Session) {
        PooledSessions::new(
            Fake {
                running: FakePart {
                    cells: POOL,
                    fed: 0,
                },
                batches: Vec::new(),
                fail_save_on: None,
                saves: 0,
            },
            DecodeSession::default(),
            max_seqs,
            sizing,
        )
        .expect("sessions")
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
        let (mut engine, first) = engine(1, Sizing::WholePool);
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
    fn a_session_dropped_after_a_failed_setup_gives_the_whole_pool_back() {
        let (mut sessions, mut first) = engine(8, Sizing::WholePool);
        sessions
            .reset_for_request(&mut first, 100, None, false)
            .expect("the first request takes the pool");
        assert_eq!(sessions.pool.free(), 0);
        // The worker drops a session whose setup failed; its part still runs in the step.
        drop(first);
        assert!(
            sessions.can_admit_session(),
            "the dropped session's cells are reclaimable"
        );
        let mut second = sessions
            .new_session()
            .expect("the next request is admitted");
        sessions
            .reset_for_request(&mut second, 100, None, false)
            .expect("and takes the whole pool");
        assert_eq!(sessions.step.context_cells(), POOL);
    }

    #[test]
    fn a_dropped_share_does_not_refuse_a_long_prompt_on_an_empty_pool() {
        let (mut sessions, mut first) = engine(2, Sizing::EvenShare);
        sessions
            .reset_for_request(&mut first, 100, None, false)
            .expect("the first request takes its share");
        assert_eq!(sessions.step.context_cells(), POOL / 2);
        drop(first);
        let mut lone = sessions.new_session().expect("admitted");
        sessions
            .reset_for_request(&mut lone, 70_000, None, false)
            .expect("a prompt above one share is served while the pool is empty");
        assert!(sessions.step.context_cells() >= 70_000);
    }

    #[test]
    fn a_live_session_keeps_its_cells() {
        let (mut sessions, mut first) = engine(8, Sizing::WholePool);
        sessions
            .reset_for_request(&mut first, 100, None, false)
            .expect("the first request takes the pool");
        assert!(!sessions.orphaned());
        assert!(matches!(
            sessions.new_session(),
            Err(EngError::MemoryPlan(_))
        ));
        assert_eq!(sessions.pool.free(), 0);
    }

    #[test]
    fn a_window_of_rows_equals_each_row_alone_through_one_batched_step() {
        let (mut engine, first) = engine(4, Sizing::EvenShare);
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
        let (mut engine, first) = engine(2, Sizing::EvenShare);
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
        let (mut engine, first) = engine(2, Sizing::EvenShare);
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
        let (mut engine, first) = engine(3, Sizing::EvenShare);
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
    fn a_row_that_has_consumed_all_but_the_last_cell_takes_one_more_token() {
        let (mut engine, first) = engine(2, Sizing::EvenShare);
        // A new session holds the initial 1024 cells; a prompt of 1023 tokens consumes 1022 and
        // leaves one pending.
        let prompt = vec![1u32; INITIAL_CELLS - 1];
        let mut a = engine.new_session().expect("a");
        engine.prefill(&mut a, &prompt).expect("prefill a");
        let mut b = first;
        engine.prefill(&mut b, &prompt).expect("prefill b");
        let (mut ga, mut gb) = (generation(8), generation(8));
        let first_step = window(&mut engine, &mut [&mut a, &mut b], &mut [&mut ga, &mut gb]);
        assert!(first_step.iter().all(Result::is_ok), "{first_step:?}");
        // Each row has now consumed 1023 of its 1024 cells with a token pending: one more, and
        // the row ends at the limit, as the solo decode does.
        let boundary = window(&mut engine, &mut [&mut a, &mut b], &mut [&mut ga, &mut gb]);
        for row in &boundary {
            let batch = row.as_ref().expect("the row takes its last cell");
            assert_eq!(batch.tokens.len(), 1);
            assert!(
                matches!(batch.finish, Some(DecodeFinish::Length)),
                "{batch:?}"
            );
        }
    }
}
