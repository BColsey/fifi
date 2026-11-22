//! The serving wrap of the resident Bolt executor.
//!
//! One session serves a request at a time; `max_seqs` above one (the default plans it from free
//! memory) serves several sessions in one decode window. The token side is the shared `DecodeSession`
//! (`family::session`): it owns the committed tokens, the pending token and
//! the finish rules (cancel, length, end of sequence, context limit). The
//! model side is a `TokenStepper` that feeds prompt rows cache-only and reads
//! the capped logits of a decode step; host sampling draws from that row.
//! Prompt rows run the cache-writing layer prefix only (the shared layers
//! after the prefix write no cache cells and feed nothing but the final norm);
//! the pending token's decode runs the head.

#[cfg(any(feature = "native-cuda", test))]
use std::path::Path;

#[cfg(any(feature = "native-cuda", test))]
use crate::EngError;

/// Whether the weights file a model path names has the geometry the Bolt engine runs. The file's
/// own header says so: no `config.json` or tokenizer file has to sit beside it. The server starter
/// uses this to dispatch inside the gemma4 family.
#[cfg(any(feature = "native-cuda", test))]
pub(crate) fn bolt_geometry(model_path: &Path) -> Result<(), EngError> {
    let (_, files) = crate::family::gguf_select::candidates(model_path)?;
    super::open_bolt_q8(&files[0].path)
}

#[cfg(feature = "native-cuda")]
fn bad(message: &str) -> EngError {
    EngError::BadRequest(format!("gemma4 serve: {message}"))
}

#[cfg(test)]
mod tests {
    use crate::family::gguf_select::candidates;

    fn scratch(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("gemma4-serve-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("scratch");
        dir
    }

    #[test]
    fn a_model_path_with_no_gguf_refuses_and_several_are_ranked() {
        let dir = scratch("none");
        assert!(candidates(&dir).is_err());
        std::fs::write(dir.join("bolt.gguf"), b"xx").expect("write");
        assert_eq!(candidates(&dir).expect("one").1.len(), 1);
        std::fs::write(dir.join("other.gguf"), b"xxxx").expect("write");
        let (_, files) = candidates(&dir).expect("two");
        assert_eq!(files[0].path, dir.join("other.gguf"));
        assert_eq!(files[1].path, dir.join("bolt.gguf"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[cfg(feature = "native-cuda")]
mod device {
    use std::path::Path;

    use super::super::budget;
    use super::super::config::MaxSeqs;
    use super::super::kv_tier::{self, BoltFamily, TIER_LABEL};
    use super::super::resident::bolt::{self, Bolt, BoltSeq, BoltSnapshot, VOCAB};
    use crate::{
        EngError,
        family::{
            cell_pool::{CellRequest, CellRoom, INITIAL_CELLS, even_share, session_cells_or_queue},
            engine_announce::{Announcement, announce_failure, await_announced},
            executor::{ExecutorOps, GenerationState},
            host_store::{HostStorePlan, largest_entry_bytes},
            kv_disk::{
                self as kv_cache, DiskOwner, DiskParts, HostStoreDisk, KvCacheSettings,
                PUMP_EMIT_BYTES,
            },
            qwen::flash_next::factory::StartedEngine,
            seq_plan::{SeqInputs, plan_seqs},
            session::{
                DecodeSession, Limits, PrefixIdentity, PrefixStore, SaveKind, ShadowLcp,
                StoreLimits, TokenStepper,
            },
            window::{self, WindowRows},
        },
        sched::{
            job::{PrefixHit, PrefixReport, PrefixTier},
            worker::{ReadyNote, spawn_engine_noted},
        },
        tokenizer::DecodeBatch,
    };

    /// The token side is the shared `DecodeSession`; `pos` is the model cache
    /// position (the number of tokens fed); `seq` holds the KV planes and position
    /// words of this session. The finish rules live in `family::session`, not here.
    pub(crate) struct BoltSession {
        core: DecodeSession,
        pos: usize,
        seq: BoltSeq,
    }

    pub(crate) struct BoltServe {
        model: Bolt,
        eos: Vec<u32>,
        /// Prefix store with no roots and one entry for each session the host funds. An entry is a copy of the saving
        /// session's cache cells below its position, held in host memory: a restore writes it
        /// into any session, so reuse works whatever the number of sessions and whichever
        /// pooled session serves the next request. Host memory, not a device slab, because the
        /// device memory belongs to the sessions' planes: an entry is at most one session's
        /// cache (952 MiB at 32,768 cells, 30,464 bytes a position) and costs a transfer over
        /// the bus (about 125 MB for a 4,000-token prefix), where recomputing the prefix costs
        /// its whole prefill.
        prefix: PrefixStore<BoltSnapshot, PrefixIdentity>,
        /// The load-time identity half of the prefix path: constant for one loaded Bolt, so
        /// live behaviour is unchanged; a persisted or cross-step store refuses entries another
        /// load wrote. Empty for now: the Bolt loader reads the GGUF header digest at open
        /// but does not retain it, and threading it means widening `bolt::load`'s return; this
        /// seam takes it when that return widens.
        identity: PrefixIdentity,
        /// The sessions the engine serves at once; the worker reads it after start.
        max_seqs: usize,
        /// The cells a request that names no generation limit may use for its whole context: the
        /// even share of the pool among the sessions.
        share: usize,
        /// The prompt rows the worker feeds per tick while sessions share the model.
        prompt_rows: usize,
        /// The kind of the entry the last restore used, the prompt tokens it shared beyond the
        /// restored prefix and the shadow lookup, for `prefix_report`.
        prefix_hit: PrefixHit,
        prefix_uncached: u32,
        prefix_shadow: Option<ShadowLcp>,
        /// The on-disk tier behind the host store, when it runs.
        disk: Option<HostStoreDisk>,
        /// The tier facts of the last restore, for `prefix_report`.
        prefix_tier: PrefixTier,
    }

    /// The store's one-tip shape (no roots, no periodic points), for the tests that drive the
    /// executor with a fixed store.
    #[cfg(test)]
    const ONE_TIP: StoreLimits = StoreLimits {
        roots: 0,
        entries: 1,
    };

    /// The model side of one session: feeds prompt rows cache-only and reads
    /// the capped logits of a decode step. The session never holds it.
    pub(crate) struct Step<'a> {
        model: &'a mut Bolt,
        seq: &'a BoltSeq,
        pos: &'a mut usize,
    }

    impl TokenStepper for Step<'_> {
        const OPERATION: &'static str = "gemma4_bolt";

        fn feed(&mut self, token: u32) -> Result<(), EngError> {
            self.model
                .prefill_batch(self.seq, &[token], *self.pos, false)?;
            *self.pos += 1;
            Ok(())
        }

        fn feed_rows(&mut self, tokens: &[u32]) -> Result<(), EngError> {
            for chunk in tokens.chunks(self.model.rows) {
                self.model
                    .prefill_batch(self.seq, chunk, *self.pos, false)?;
                *self.pos += chunk.len();
            }
            Ok(())
        }

        fn logits(&mut self, token: u32) -> Result<Vec<f32>, EngError> {
            self.model.decode(self.seq, token, *self.pos)?;
            *self.pos += 1;
            self.model.capped_logits()
        }

        /// The next token is drawn on the device from the capped logits: one word leaves the card.
        fn next_token(
            &mut self,
            token: u32,
            sampler: &mut crate::family::sampler::Sampler,
            sampler_vocab: usize,
        ) -> Result<u32, EngError> {
            if sampler_vocab != VOCAB {
                return Err(EngError::Backend {
                    operation: Self::OPERATION,
                    message: "the sampler vocabulary is not the head width".to_owned(),
                });
            }
            if sampler.capture_armed() {
                let logits = self.logits(token)?;
                return crate::family::decode_step::draw::host_draw(
                    Self::OPERATION,
                    &logits,
                    sampler,
                    sampler_vocab,
                );
            }
            let drawn = self
                .model
                .decode_draw(self.seq, token, *self.pos, sampler)?;
            *self.pos += 1;
            sampler.record_device_draw(drawn);
            Ok(drawn)
        }

        fn fed(&self) -> usize {
            *self.pos
        }

        fn reset(&mut self) -> Result<(), EngError> {
            // A replay from position 0 overwrites every cache cell it uses.
            *self.pos = 0;
            Ok(())
        }
    }

    impl ExecutorOps for BoltServe {
        type Session = BoltSession;

        fn max_seqs(&self) -> usize {
            self.max_seqs
        }

        /// Bolt's batching is measured: idle prompts take whole pieces, and a pooled session that
        /// idles for the default interval is released.
        fn batch_policy(&self) -> crate::family::executor::BatchPolicy {
            budget::bolt_batch_policy()
        }

        /// Whether the device has room for one more session's planes and the headroom. A probe
        /// that cannot read the device says yes, and the allocation decides.
        fn can_admit_session(&self) -> bool {
            if self.model.pool().free() < INITIAL_CELLS {
                return false;
            }
            let Some(cost) = budget::bolt_seq_cost() else {
                return true;
            };
            let per_session = cost
                .per_position
                .saturating_mul(INITIAL_CELLS as u64)
                .saturating_add(cost.fixed)
                .saturating_add(budget::BOLT_HEADROOM_BYTES);
            self.model
                .ctx
                .memory_info()
                .map_or(true, |info| info.free as u64 >= per_session)
        }

        /// The entries live in host memory and the sessions on the device: dropping an entry does
        /// not make room for a session.
        fn eviction_frees_session_memory(&self) -> bool {
            false
        }

        fn prompt_rows_per_tick(&self) -> Option<usize> {
            (self.max_seqs > 1).then_some(self.prompt_rows)
        }

        fn new_session(&mut self) -> Result<Self::Session, EngError> {
            Ok(BoltSession {
                core: DecodeSession::default(),
                pos: 0,
                seq: self.model.new_seq_cells(INITIAL_CELLS)?,
            })
        }

        /// Sizes the session's cells to the request, leased from the pool: a request that names
        /// its generation limit holds the prompt and the limit, one that names none holds the
        /// even share, and a prompt above the share holds what it needs while the pool can
        /// fund it. A pool that cannot fund the prompt yet refuses with a memory-plan error and
        /// the worker queues the request until a session lets go. The planes are released
        /// before the new ones are allocated and their contents are not kept: this runs before
        /// any prefix restore.
        fn reset_for_request(
            &mut self,
            st: &mut Self::Session,
            prompt_tokens: usize,
            completion_limit: Option<u32>,
            _has_prefix_boundary: bool,
        ) -> Result<(), EngError> {
            let room = CellRoom {
                reachable: st.seq.reachable(),
                ceiling: self.model.cells(),
                share: self.share,
            };
            let request = CellRequest {
                prompt: prompt_tokens,
                completion: completion_limit,
            };
            let cells = session_cells_or_queue(request, room)?;
            self.model.resize_seq(&mut st.seq, cells)?;
            self.reset(st)
        }

        fn decode_window(
            &mut self,
            sessions: &mut [&mut Self::Session],
            generations: &mut [&mut GenerationState],
        ) -> Result<Vec<Result<crate::tokenizer::DecodeBatch, EngError>>, EngError> {
            window::decode_window(self, sessions, generations)
        }

        fn history<'s>(&self, st: &'s Self::Session) -> &'s [u32] {
            st.core.history()
        }

        fn reset(&mut self, st: &mut Self::Session) -> Result<(), EngError> {
            // The restore facts of the report do not survive a job.
            self.prefix_hit = PrefixHit::Miss;
            self.prefix_uncached = 0;
            self.prefix_shadow = None;
            self.prefix_tier = PrefixTier::default();
            st.core.reset(&mut Step {
                model: &mut self.model,
                seq: &st.seq,
                pos: &mut st.pos,
            })
        }

        fn check_prompt(&self, prompt_tokens: usize) -> Result<(), EngError> {
            // The most one session holds: a prompt below it queues until the pool funds it.
            crate::family::session::check_prompt(prompt_tokens, self.model.cells())
        }

        /// The limits of the store itself, so the worker's save-point
        /// planning sees what the store really keeps.
        fn prefix_limits(&self) -> StoreLimits {
            self.prefix.limits()
        }

        fn prefill(&mut self, st: &mut Self::Session, toks: &[u32]) -> Result<(), EngError> {
            let cells = st.seq.cells();
            st.core.prefill(
                &mut Step {
                    model: &mut self.model,
                    seq: &st.seq,
                    pos: &mut st.pos,
                },
                toks,
                cells,
            )
        }

        fn decode(
            &mut self,
            st: &mut Self::Session,
            generation: &mut GenerationState,
        ) -> Result<DecodeBatch, EngError> {
            let limits = Limits {
                eos: &self.eos,
                sampler_vocab: VOCAB,
                context_limit: st.seq.cells(),
            };
            st.core.decode(
                &mut Step {
                    model: &mut self.model,
                    seq: &st.seq,
                    pos: &mut st.pos,
                },
                generation,
                &limits,
            )
        }

        fn boundary(&mut self, _st: &mut Self::Session) {}

        /// Saves a copy of the session's cache cells below its position as an entry of the
        /// store. The session rules (the pending token, the strict key match) are
        /// `DecodeSession`'s; the model state at the save point is the cells `0..pos`. A host
        /// that cannot hold the copy refuses with a memory-plan error, which the store treats as
        /// a denied save.
        fn save_prefix(
            &mut self,
            st: &Self::Session,
            key: &[u32],
            kind: SaveKind,
        ) -> Result<bool, EngError> {
            let identity = self.identity.clone();
            let model = &self.model;
            let saved =
                st.core
                    .save_prefix(&mut self.prefix, key, identity, st.pos, kind, || {
                        model.snapshot_seq(&st.seq, st.pos)
                    })?;
            if saved {
                self.spill_after_save();
            }
            Ok(saved)
        }

        /// Writes the saved entry into the (reset) session and moves its position to the entry's.
        /// The cells above stay as they are; every read is bounded by the position word and the
        /// tail prefill that follows overwrites them. Any session can take any entry.
        fn restore_prefix(
            &mut self,
            st: &mut Self::Session,
            prompt: &[u32],
        ) -> Result<usize, EngError> {
            // The host store missed or holds a shorter prefix: the disk may hold a longer one.
            self.promote_from_disk((), prompt);
            let identity = self.identity.clone();
            let model = &self.model;
            let BoltSession { core, pos, seq } = st;
            let covered = core.restore_prefix(&mut self.prefix, prompt, identity, |saved| {
                model.restore_seq(seq, saved)?;
                *pos = saved.fed();
                Ok(())
            })?;
            // The restore facts of the report, read now: the store moves on with the job's saves.
            self.prefix_hit = self.prefix.restored_kind().into();
            self.prefix_uncached = self.prefix.uncached_shared(prompt, covered);
            self.prefix_shadow = self.prefix.shadow_report();
            self.prefix_tier = self.restore_tier_of(covered, None);
            Ok(covered)
        }

        /// The entries the store holds, the host bytes of their copies, and the facts of the
        /// last restore, for the request log and the usage report.
        fn prefix_report(&self) -> Option<PrefixReport> {
            let stored = self.prefix.roots_len() + self.prefix.entries_len();
            Some(PrefixReport {
                entries: u32::try_from(stored).unwrap_or(u32::MAX),
                bytes: self
                    .prefix
                    .snapshots()
                    .map(|snapshot| snapshot.bytes() as u64)
                    .sum(),
                hit: self.prefix_hit,
                uncached_shared: self.prefix_uncached,
                shadow: self.prefix_shadow,
                restore_drops: self.prefix.restore_drop_count(),
                evictions: self.prefix.eviction_count(),
                save_side: self.prefix.save_side_totals(),
                tier: self.prefix_tier,
            })
        }

        /// Whether the disk tier has saves the engine thread should advance while it waits.
        fn background_pending(&self) -> bool {
            DiskOwner::disk_pending(self)
        }

        /// Copies a bounded part of the pending saves to the writer: a few MiB per call, which a
        /// batching worker makes between ticks and an idle one every 2 ms.
        fn background_step(&mut self) {
            self.pump_disk(PUMP_EMIT_BYTES);
        }

        /// A graceful stop: writes the tips that exist only in host memory, within the tier's
        /// bound.
        fn background_drain(&mut self) {
            self.drain_disk();
        }
    }

    impl DiskOwner for BoltServe {
        type Snapshot = BoltSnapshot;
        type Path = PrefixIdentity;
        type Family<'a> = BoltFamily<'a>;
        type Restore = ();

        fn disk_ref(&self) -> Option<&HostStoreDisk> {
            self.disk.as_ref()
        }

        fn disk_mut(&mut self) -> Option<&mut HostStoreDisk> {
            self.disk.as_mut()
        }

        fn disk_parts(&mut self, _restore: Option<()>) -> Option<DiskParts<'_, BoltFamily<'_>>> {
            let model = &self.model;
            let disk = self.disk.as_mut()?;
            Some(DiskParts {
                disk,
                family: BoltFamily {
                    layer_bytes: Box::new(|cells| model.snapshot_layer_bytes(cells)),
                    context_cells: model.cells(),
                },
                store: &mut self.prefix,
            })
        }

        fn restore_path(&self) -> PrefixIdentity {
            self.identity.clone()
        }
    }

    impl WindowRows for BoltServe {
        type Stepper<'a> = Step<'a>;
        /// The planes are preallocated, so a position costs no growth.
        type Permit = ();
        const OPERATION: &'static str = "gemma4_bolt";

        fn split<'a>(
            &'a mut self,
            session: &'a mut BoltSession,
        ) -> (&'a mut DecodeSession, Step<'a>, Limits<'a>) {
            let BoltSession { core, pos, seq } = session;
            let cells = seq.cells();
            (
                core,
                Step {
                    model: &mut self.model,
                    seq,
                    pos,
                },
                Limits {
                    eos: &self.eos,
                    sampler_vocab: VOCAB,
                    context_limit: cells,
                },
            )
        }

        fn check_row(&self, session: &BoltSession, token: u32) -> Result<(), EngError> {
            let cells = session.seq.cells();
            if session.pos >= cells {
                return Err(EngError::ContextLimit(format!(
                    "position {} reaches the {cells} cell context",
                    session.pos
                )));
            }
            if token as usize >= VOCAB {
                return Err(EngError::BadRequest(
                    "token is outside the embedding table".to_owned(),
                ));
            }
            Ok(())
        }

        fn row_growth(&self, _session: &BoltSession) -> Result<u64, EngError> {
            Ok(0)
        }

        fn reserve_growth(&self, _bytes: u64) -> Result<(), EngError> {
            Ok(())
        }

        /// The logits rows of a window with a request that records its draw: only the host walk
        /// has the candidates.
        fn step_rows(
            &mut self,
            sessions: &mut [&mut BoltSession],
            tokens: &[u32],
        ) -> Result<Vec<Vec<f32>>, EngError> {
            let seqs: Vec<&BoltSeq> = sessions.iter().map(|session| &session.seq).collect();
            let positions: Vec<usize> = sessions.iter().map(|session| session.pos).collect();
            let rows = self.model.decode_rows(&seqs, tokens, &positions)?;
            for session in sessions.iter_mut() {
                session.pos += 1;
            }
            Ok(rows)
        }

        fn draws_on_device(&self) -> bool {
            true
        }

        /// One token of every session through one step, and the next token of each drawn from
        /// its own sampler.
        fn step_rows_drawn(
            &mut self,
            sessions: &mut [&mut BoltSession],
            tokens: &[u32],
            samplers: &[&crate::family::sampler::Sampler],
        ) -> Result<Vec<u32>, EngError> {
            let seqs: Vec<&BoltSeq> = sessions.iter().map(|session| &session.seq).collect();
            let positions: Vec<usize> = sessions.iter().map(|session| session.pos).collect();
            let drawn = self
                .model
                .decode_rows_draw(&seqs, tokens, &positions, samplers)?;
            for session in sessions.iter_mut() {
                session.pos += 1;
            }
            Ok(drawn)
        }
    }

    /// The free device bytes the planner may spend on extra sessions: what remains once the
    /// first session is allocated.
    fn plan_sessions(
        model: &Bolt,
        seqs: MaxSeqs,
    ) -> Result<crate::family::seq_plan::SeqPlan, EngError> {
        let free = u64::try_from(model.ctx.memory_info()?.free).unwrap_or(u64::MAX);
        let inputs = SeqInputs {
            free,
            headroom: budget::BOLT_HEADROOM_BYTES,
            cost: budget::bolt_seq_cost()
                .ok_or_else(|| EngError::MemoryPlan("the Bolt sequence cost overflows".into()))?,
            context: INITIAL_CELLS as u64,
            row_cap: budget::BOLT_WINDOW_SEQS,
            ceiling: budget::BOLT_WINDOW_SEQS.min((model.pool().total() / INITIAL_CELLS).max(1)),
        };
        plan_seqs(
            inputs,
            match seqs {
                MaxSeqs::Count(count) => Some(count),
                MaxSeqs::Auto => None,
            },
        )
    }

    /// The prefix store the start gives the executor, and the plan behind it. It holds host copies
    /// of up to a whole session's cache, one entry for each conversation the engine serves at
    /// once, as many as the host funds. A configuration that turns reuse off keeps nothing. No
    /// roots: a prompt has no stable head to pin. `host` reads the host facts and runs only when
    /// reuse is on.
    fn prefix_store(
        prefix_reuse: bool,
        budget: impl FnOnce() -> Option<u64>,
        sessions: usize,
        cells: usize,
    ) -> (PrefixStore<BoltSnapshot, PrefixIdentity>, HostStorePlan) {
        let entry_bytes = largest_entry_bytes(budget::bolt_kv_bytes_per_position(), cells as u64);
        let plan = HostStorePlan::for_start(prefix_reuse, entry_bytes, budget, sessions);
        (PrefixStore::with_limits(plan.limits(0)), plan)
    }

    /// `model_path` is the model directory or one `.gguf` file in it. A directory with several
    /// `.gguf` files serves the highest-precision one that the device memory admits; `seqs`
    /// is the explicit session count or the count the free memory funds. Several sessions turn
    /// the decode graph off, and the start line says so; prefix reuse works at every count.
    pub(crate) fn start_bolt_engine(
        model_path: &Path,
        tokenizer_dir: &Path,
        ready_note: ReadyNote,
        seqs: MaxSeqs,
        prompt_rows: Option<usize>,
        kv_cache: Option<KvCacheSettings>,
        prefix_reuse: bool,
    ) -> Result<StartedEngine, EngError> {
        let tier_plan = kv_cache::plan_tier(kv_cache.as_ref(), "gemma4", TIER_LABEL)?;
        let (_, files) = crate::family::gguf_select::candidates(model_path)?;
        let bundle = crate::tokenizer::gemma::load(tokenizer_dir)?;
        let text_identity = *bundle.identity();
        let eos = bundle.eos_token_ids().to_vec();
        if eos.is_empty() || eos.iter().any(|&id| id as usize >= VOCAB) {
            return Err(super::bad(
                "EOS ids must be non-empty and within vocabulary",
            ));
        }
        if bundle.vocab_size() as usize != VOCAB {
            return Err(super::bad(
                "the tokenizer vocabulary differs from the checkpoint vocabulary",
            ));
        }
        // The engine thread plans the KV cells from the free card and announces them before the
        // weights are uploaded; the server's request bound needs the figure before the engine is
        // ready.
        let (announce, resolved) = std::sync::mpsc::sync_channel::<Announcement<usize>>(1);
        let engine = spawn_engine_noted(budget::BOLT_PREFILL_LADDER[0], ready_note, move |_| {
            let loaded = bolt::load_choosing(&files, seqs, &|cells| {
                let _ = announce.try_send(Ok(cells));
            });
            announce_failure(&loaded, Some(&announce));
            let mut model = loaded?;
            let first = model.new_seq_cells(INITIAL_CELLS)?;
            let ask = budget::session_ask(seqs, model.graph_mode.active());
            if let Some(note) = &ask.note {
                eprintln!("gemma4 bolt: {note}");
            }
            let plan = plan_sessions(&model, ask.seqs)?;
            eprintln!("gemma4 bolt: {}", plan.describe());
            if plan.seqs == 1 && prompt_rows.is_some() {
                eprintln!(
                    "gemma4 bolt: prompt_rows has no effect with one session: it sets how many prompt rows are fed per tick while several sessions share the model"
                );
            }
            if plan.seqs > 1 {
                model.reserve_window(plan.seqs)?;
                model.use_decode_graph(false);
                eprintln!(
                    "gemma4 bolt: {} sessions share the model, so the decode graph is off",
                    plan.seqs
                );
            }
            // What names this load to an entry kept outside the process: the checkpoint, the
            // pinned tokenizer and template, the layout of a snapshot and the numeric path.
            let load = kv_tier::load_identity(
                &model.weights_identity,
                &text_identity,
                &model.snapshot_layer_bytes(1),
                bolt::attn_split_wanted()?.then(|| model.attn_splits()),
            );
            let identity = PrefixIdentity {
                weights: load.weights.as_str().into(),
                tokenizer: load.tokenizer.as_str().into(),
                template: load.template.as_str().into(),
            };
            let (prefix, store) = prefix_store(
                prefix_reuse,
                || kv_cache::default_host_budget(crate::family::mem::process_resident_bytes()),
                plan.seqs,
                model.cells(),
            );
            eprintln!("gemma4 bolt: {}", store.describe());
            let disk = tier_plan.as_ref().and_then(|(resolved, origin)| {
                HostStoreDisk::open(TIER_LABEL, resolved, origin, &load, store.entries.max(1))
            });
            let model_pool = model.pool().total();
            let mut serve = BoltServe {
                prompt_rows: prompt_rows.unwrap_or(model.rows),
                model,
                eos: eos.clone(),
                prefix,
                identity,
                max_seqs: plan.seqs,
                share: even_share(model_pool, plan.seqs),
                prefix_hit: PrefixHit::Miss,
                prefix_uncached: 0,
                prefix_shadow: None,
                disk,
                prefix_tier: PrefixTier::default(),
            };
            serve.log_host_budget();
            if plan.seqs > 1 {
                eprintln!(
                    "gemma4 bolt: {} prompt rows per tick while sessions share the model ({})",
                    serve.prompt_rows,
                    if prompt_rows.is_some() {
                        "set"
                    } else {
                        "automatic"
                    }
                );
            }
            Ok((
                serve,
                BoltSession {
                    core: DecodeSession::default(),
                    pos: 0,
                    seq: first,
                },
            ))
        })?;
        let cells = await_announced(&resolved, "Bolt", || engine.handle().failure())?;
        Ok(StartedEngine {
            engine,
            bundle,
            context_limit: u32::try_from(cells)
                .map_err(|_| EngError::MemoryPlan("the Bolt cell count exceeds u32".into()))?,
        })
    }

    #[cfg(test)]
    mod prefix_tests {
        use super::*;

        const GIB: u64 = 1 << 30;

        /// The start builds its store through `prefix_store`, so these cases are what a start does
        /// with the flag, the host and the session count.
        #[test]
        fn the_start_gives_the_store_the_flag_the_sessions_and_the_host() {
            let roomy = || Some(48 * GIB);
            // Reuse off keeps no entries and never reads the host.
            let (store, plan) = prefix_store(false, || panic!("the host is not read"), 4, 32_768);
            assert_eq!((store.limits().roots, store.limits().entries), (0, 0));
            assert!(plan.configured_off);
            // Reuse on keeps one entry for each session, and no roots.
            for sessions in [1, 2, 4, 8] {
                let (store, _) = prefix_store(true, roomy, sessions, 32_768);
                assert_eq!(
                    (store.limits().roots, store.limits().entries),
                    (0, sessions),
                    "{sessions} sessions"
                );
            }
            // A budget that funds two entries bounds a larger engine.
            let two = || Some(2 * 952 * (1 << 20));
            assert_eq!(prefix_store(true, two, 8, 32_768).0.limits().entries, 2);
            // A budget under one entry keeps the store planned, with its saves waiting on the host.
            let busy = || Some(GIB / 2);
            let (store, plan) = prefix_store(true, busy, 4, 32_768);
            assert_eq!(store.limits().entries, 1);
            assert!(!plan.funded && plan.on());
            // A machine whose memory is unknown keeps one entry.
            assert_eq!(prefix_store(true, || None, 4, 32_768).0.limits().entries, 1);
        }

        /// Panics when no Bolt Q8_0 file is configured: an explicit
        /// `--ignored` run without `BOLT_Q8_FILE` is a setup error, not a pass.
        fn bolt_file() -> std::path::PathBuf {
            std::env::var("BOLT_Q8_FILE")
                .map(std::path::PathBuf::from)
                .expect("BOLT_Q8_FILE must point at a Bolt Q8_0 file")
        }

        /// Greedy argmax of the capped logits after one decode step.
        fn greedy_step(model: &mut Bolt, seq: &BoltSeq, token: u32, pos: usize) -> u32 {
            model.decode(seq, token, pos).expect("decode");
            let logits = model.capped_logits().expect("capped logits");
            logits
                .iter()
                .enumerate()
                .fold((0u32, logits[0]), |(best, value), (index, &other)| {
                    if other > value {
                        (index as u32, other)
                    } else {
                        (best, value)
                    }
                })
                .0
        }

        /// Feeds a prompt cache-only, holding nothing pending: the model-level
        /// control for cold runs.
        fn feed(model: &mut Bolt, seq: &BoltSeq, prompt: &[u32]) {
            for (index, &token) in prompt.iter().enumerate() {
                model
                    .prefill_batch(seq, &[token], index, false)
                    .expect("prefill");
            }
        }

        /// A true cold replay of a restore past a decode. Turn 1 decodes 48
        /// greedy tokens past the save point (cells P..P+48 now hold turn-1
        /// tokens); the restore rewinds to P, the tail prefill overwrites only
        /// the first cells of that span with DIFFERENT tokens, and the drawn
        /// chain must equal the chain a cold run of prompt + tail draws after a
        /// reset. A cache read above the position word (or a ring-buffer
        /// window) would leak the surviving turn-1 cells into the restored run
        /// and break the equality. The prompt is 3,000 tokens so turn 1 crosses
        /// the 512-cell sliding window.
        #[test]
        #[ignore = "requires a Bolt Q8_0 file and a device"]
        fn a_restore_past_a_decode_matches_the_cold_replay() {
            let mut model: Bolt = bolt::load(&bolt_file()).expect("load");
            let seq = model.new_seq().expect("sequence");
            let prompt: Vec<u32> = (0..3_000u32).map(|index| 100 + (index % 977)).collect();
            feed(&mut model, &seq, &prompt);
            // The save point mirrors the session's: `prompt.len() - 1` tokens
            // fed, the last token held pending.
            let saved = prompt.len() - 1;
            // Turn 1: 48 greedy tokens past the save point.
            let mut token = prompt[saved];
            for pos in (saved..).take(48) {
                token = greedy_step(&mut model, &seq, token, pos);
            }
            // The restored branch: rewind to the save point, feed the held
            // token plus an extension whose tokens differ from turn 1's drawn
            // tokens (one window of rows), then draw greedily.
            let extension: Vec<u32> = (0..7).map(|index| 90_000 + index).collect();
            let mut tail: Vec<u32> = vec![prompt[saved]];
            tail.extend_from_slice(&extension);
            let draw = |model: &mut Bolt| -> Vec<u32> {
                model
                    .prefill_batch(&seq, &tail, saved, false)
                    .expect("tail prefill");
                let mut token = *tail.last().expect("tail");
                let mut drawn = Vec::new();
                for pos in (saved + tail.len()..).take(24) {
                    token = greedy_step(model, &seq, token, pos);
                    drawn.push(token);
                }
                drawn
            };
            let restored = draw(&mut model);
            // The cold control: reset (position 0), re-feed the prompt, then
            // the same tail and greedy continuation.
            feed(&mut model, &seq, &prompt);
            let cold = draw(&mut model);
            assert_eq!(restored, cold, "the restore must match the cold replay");
            assert!(
                restored.windows(2).any(|pair| pair[0] != pair[1]),
                "the drawn chain must not be constant"
            );
        }

        /// The graphed decode must draw the identical greedy chain as the
        /// eager one. Run 1 draws eagerly; the re-feed from position 0
        /// overwrites every cell the first run read below its position word;
        /// run 2 arms the graph (the first clean step warms, the second
        /// captures, the rest replay) and must reproduce run 1's chain over a
        /// mid-run prefill too — the restore flow feeds rows while the graph
        /// exists, and the next decode must still replay it.
        #[test]
        #[ignore = "requires a Bolt Q8_0 file and a device"]
        fn a_graphed_greedy_run_matches_the_eager_run() {
            let mut model: Bolt = bolt::load(&bolt_file()).expect("load");
            let seq = model.new_seq().expect("sequence");
            assert!(
                !model.graph_mode.active(),
                "the graph stays off without FIFI_BOLT_DECODE_GRAPH"
            );
            // 1,000 prompt rows and 96 more drawn: every decode reads past
            // the 512-cell sliding window of the local layers (Bolt's
            // LOCAL_WINDOW is the spec's sliding_window), so the windowed
            // attention of a replay is exercised over its full span.
            let prompt: Vec<u32> = (0..1_000u32).map(|index| 100 + (index % 977)).collect();
            let extension: Vec<u32> = (0..8u32).map(|index| 90_000 + index).collect();
            let run = |model: &mut Bolt, prompt: &[u32]| -> Vec<u32> {
                feed(model, &seq, prompt);
                let saved = prompt.len() - 1;
                let mut token = prompt[saved];
                let mut drawn = Vec::new();
                for pos in (saved..).take(10) {
                    token = greedy_step(model, &seq, token, pos);
                    drawn.push(token);
                }
                model
                    .prefill_batch(&seq, &extension, saved + 10, false)
                    .expect("mid-run prefill");
                let mut token = *extension.last().expect("extension");
                for pos in (saved + 10 + extension.len()..).take(86) {
                    token = greedy_step(model, &seq, token, pos);
                    drawn.push(token);
                }
                drawn
            };
            let eager = run(&mut model, &prompt);
            // Poison: a divergent eager run leaves OTHER values in every cell
            // the graphed run will read, so a replay that reads or writes a
            // stale cell (the capture-time position, say) draws a different
            // chain instead of matching the identical leftovers.
            let poison: Vec<u32> = (0..1_000u32).map(|index| 200_000 + (index % 891)).collect();
            let spoiled = run(&mut model, &poison);
            assert_ne!(eager, spoiled, "the poison run must diverge");
            model.use_decode_graph(true);
            let graphed = run(&mut model, &prompt);
            assert_eq!(eager, graphed, "the graphed chain must be byte-identical");
            assert!(
                model.graph_mode.replays() > 80,
                "the steps must have replayed (replays: {})",
                model.graph_mode.replays()
            );
            assert_eq!(model.graph_mode.refusal(), None, "no refusal may occur");
            let mut status = [0u8; 4];
            model
                .ctx
                .download(&model.scratch.status, &mut status)
                .expect("status word");
            assert_eq!(
                u32::from_le_bytes(status),
                0,
                "no KV8 append may have refused mid-run"
            );
            assert!(
                eager.windows(2).any(|pair| pair[0] != pair[1]),
                "the drawn chain must not be constant"
            );
        }

        /// The graphed decode with the split attention armed must still draw
        /// the identical greedy chain as the eager split run: the capture
        /// replays the part+combine pair exactly. The same poison discipline
        /// as the unsplit parity test, at a shorter draw to keep the suite
        /// time bounded.
        #[test]
        #[ignore = "requires a Bolt Q8_0 file and a device"]
        fn a_graphed_split_run_matches_the_eager_split_run() {
            let mut model: Bolt = bolt::load(&bolt_file()).expect("load");
            let seq = model.new_seq().expect("sequence");
            let prompt: Vec<u32> = (0..1_000u32).map(|index| 100 + (index % 977)).collect();
            // The chain AND the logits bits per step: matching the greedy
            // chain alone is weaker than the pin this test owes, and the
            // bits make every replayed buffer write checkable.
            let run = |model: &mut Bolt, prompt: &[u32]| -> (Vec<u32>, Vec<Vec<u32>>) {
                feed(model, &seq, prompt);
                let saved = prompt.len() - 1;
                let mut token = prompt[saved];
                let mut drawn = Vec::new();
                let mut bits = Vec::new();
                for pos in (saved..).take(24) {
                    model.decode(&seq, token, pos).expect("decode");
                    let logits = model.capped_logits().expect("capped logits");
                    bits.push(logits.iter().map(|value| value.to_bits()).collect());
                    token = logits
                        .iter()
                        .enumerate()
                        .fold((0u32, logits[0]), |(best, value), (index, &other)| {
                            if other > value {
                                (index as u32, other)
                            } else {
                                (best, value)
                            }
                        })
                        .0;
                    drawn.push(token);
                }
                (drawn, bits)
            };
            // Capture a decode graph against the unsplit state first.
            model.use_decode_graph(true);
            let (unsplit, _) = run(&mut model, &prompt);
            assert!(model.graph_captured(), "the graph must be captured");
            let replays_unsplit = model.graph_mode.replays();
            // Arming the split must drop that graph: its capture points at the
            // pre-split launches, and a replay after the part buffer appears
            // (or the buffer frees on disarm) would read freed or dead state.
            model.use_attn_split(true).expect("arm split");
            assert!(!model.graph_captured(), "arming must invalidate the graph");
            model.use_decode_graph(false);
            let (eager, eager_bits) = run(&mut model, &prompt);
            assert!(
                eager.windows(2).any(|pair| pair[0] != pair[1]),
                "the drawn chain must not be constant"
            );
            let poison: Vec<u32> = (0..1_000u32).map(|index| 200_000 + (index % 891)).collect();
            let (spoiled, _) = run(&mut model, &poison);
            assert_ne!(eager, spoiled, "the poison run must diverge");
            model.use_attn_split(false).expect("disarm split");
            assert!(
                !model.split_armed(),
                "the disarm itself must free the partial buffer"
            );
            model.use_attn_split(true).expect("re-arm split");
            model.use_decode_graph(true);
            let (graphed, graphed_bits) = run(&mut model, &prompt);
            assert_eq!(eager, graphed, "the graphed split chain must match");
            assert_eq!(
                eager_bits, graphed_bits,
                "every replayed logits word must be bit-identical to the eager split run"
            );
            assert!(
                model.graph_captured(),
                "the graph must have recaptured after the re-arm"
            );
            assert!(
                model.graph_mode.replays() > replays_unsplit + 20,
                "the recaptured graph must replay the split steps (delta {})",
                model.graph_mode.replays() - replays_unsplit
            );
            assert!(model.graph_mode.replays() > 20, "must have replayed");
            assert_eq!(model.graph_mode.refusal(), None, "no refusal may occur");
            let _ = unsplit;
        }

        /// A serve with room for three sessions and no tokenizer: the tests drive the executor
        /// seam and the model directly.
        fn serve_of(model: Bolt) -> BoltServe {
            serve_with(model, 3)
        }

        fn serve_with(model: Bolt, sessions: usize) -> BoltServe {
            let pool = model.pool().total();
            BoltServe {
                prompt_rows: model.rows,
                model,
                eos: vec![0],
                prefix: PrefixStore::with_limits(ONE_TIP),
                identity: PrefixIdentity::default(),
                max_seqs: sessions,
                share: even_share(pool, sessions),
                prefix_hit: PrefixHit::Miss,
                prefix_uncached: 0,
                prefix_shadow: None,
                disk: None,
                prefix_tier: PrefixTier::default(),
            }
        }

        /// A new session sized to a request of `prompt_tokens` that names no generation limit.
        fn sized_session(serve: &mut BoltServe, prompt_tokens: usize) -> BoltSession {
            let mut session = serve.new_session().expect("session");
            serve
                .reset_for_request(&mut session, prompt_tokens, None, false)
                .expect("sized for the request");
            session
        }

        /// One prompt near the model's context takes the whole pool of a default start; eight
        /// short sessions take an even share each and a long prompt queues while they are live.
        /// The cells of every session are printed.
        #[test]
        #[ignore = "requires the pinned Bolt file and a device; use --release --test-threads=1"]
        fn one_long_prompt_or_eight_short_sessions_share_the_pool() {
            let path = pinned_bolt();
            let bytes = std::fs::metadata(&path).expect("model file").len();
            let model = bolt::load_choosing(
                &[crate::family::gguf_select::GgufFile { path, bytes }],
                None,
                MaxSeqs::Auto,
                &|_| {},
            )
            .expect("load");
            let (pool, ceiling) = (model.pool().total(), model.cells());
            let mut serve = serve_with(model, Bolt::WINDOW_ROWS);
            let share = serve.share;
            eprintln!("pool {pool} cells, one session up to {ceiling}, even share {share}");

            // Eight short sessions run together, each at its even share.
            let short = prompt(40, 1);
            let mut sessions: Vec<BoltSession> = (0..Bolt::WINDOW_ROWS)
                .map(|_| sized_session(&mut serve, short.len()))
                .collect();
            for session in &sessions {
                assert_eq!(session.seq.cells(), share);
            }
            eprintln!(
                "8 short sessions: {:?} cells each, {} pool cells free",
                sessions.iter().map(|s| s.seq.cells()).collect::<Vec<_>>(),
                serve.model.pool().free()
            );
            for session in &mut sessions {
                serve.prefill(session, &short).expect("prefill");
            }
            let seqs: Vec<&BoltSeq> = sessions.iter().map(|session| &session.seq).collect();
            serve.model.reserve_window(seqs.len()).expect("window rows");
            let positions: Vec<usize> = sessions.iter().map(|session| session.pos).collect();
            let rows = serve
                .model
                .decode_rows(&seqs, &vec![2u32; seqs.len()], &positions)
                .expect("eight sessions decode together");
            assert_eq!(rows.len(), Bolt::WINDOW_ROWS);

            // A prompt above an even share queues while they are live: the pool cannot fund it.
            let long_len = ceiling - 4_096;
            let mut waiting = serve.new_session().expect("a ninth session");
            assert!(matches!(
                serve.reset_for_request(&mut waiting, long_len, None, false),
                Err(EngError::MemoryPlan(_))
            ));
            drop(waiting);

            // The sessions let go and the same prompt takes the pool.
            drop(sessions);
            let mut lone = sized_session(&mut serve, long_len);
            eprintln!(
                "1 long prompt of {long_len} tokens: {} cells, {} pool cells free",
                lone.seq.cells(),
                serve.model.pool().free()
            );
            assert_eq!(lone.seq.cells(), ceiling);
            let started = std::time::Instant::now();
            serve
                .prefill(&mut lone, &prompt(long_len, 3))
                .expect("the long prompt prefills");
            eprintln!("prefill of {long_len} tokens took {:?}", started.elapsed());
            // The last prompt token stays pending for the first decode step.
            assert_eq!(lone.pos, long_len - 1);
        }

        fn prompt(len: usize, salt: u32) -> Vec<u32> {
            (0..len as u32)
                .map(|index| 100 + salt + (index % 977))
                .collect()
        }

        /// The capped logits bits of decoding the pending last token of `session`.
        fn next_logits(serve: &mut BoltServe, session: &BoltSession, token: u32) -> Vec<u32> {
            serve
                .model
                .decode(&session.seq, token, session.pos)
                .expect("decode");
            serve
                .model
                .capped_logits()
                .expect("logits")
                .iter()
                .map(|value| value.to_bits())
                .collect()
        }

        /// A prefix saved by one session restores into a different, dirty session and the
        /// continuation matches a cold run bit for bit, with the source session having moved on
        /// (decoded and reset) since the save.
        #[test]
        #[ignore = "requires a Bolt Q8_0 file and a device"]
        fn a_saved_prefix_restores_into_another_session_and_matches_a_cold_run() {
            restore_matches_a_cold_run(300);
        }

        /// The same past the 512-cell sliding window of the local layers and with planes larger
        /// than the 8 MiB zero chunk (20,000 positions: 10 MB per global plane range, 5 MB per
        /// local one). The largest range a snapshot can copy is 16 MB (32,768 cells of 512), under
        /// the 64 MiB staging chunk of a device copy, so one copy never splits a range.
        #[test]
        #[ignore = "requires a Bolt Q8_0 file and a device"]
        fn a_long_saved_prefix_restores_bit_exact_past_the_window_and_the_copy_chunk() {
            restore_matches_a_cold_run(20_000);
        }

        fn restore_matches_a_cold_run(base_len: usize) {
            let mut serve = serve_of(bolt::load(&bolt_file()).expect("load"));
            let base = prompt(base_len, 0);
            let extended: Vec<u32> = base.iter().copied().chain(prompt(40, 5_000)).collect();
            let mut source = sized_session(&mut serve, base.len());
            let mut target = sized_session(&mut serve, extended.len());
            let mut cold = sized_session(&mut serve, extended.len());

            serve.reset(&mut source).expect("reset");
            serve.prefill(&mut source, &base).expect("prefill");
            assert!(
                serve
                    .save_prefix(&source, &base, SaveKind::Tip)
                    .expect("save")
            );
            // The source moves on: it decodes past the save point and is reset for another job.
            let mut token = *base.last().expect("prompt");
            for step in 0..8 {
                serve
                    .model
                    .decode(&source.seq, token, source.pos + step)
                    .expect("decode");
                token = 200 + step as u32;
            }
            serve.reset(&mut source).expect("reset");
            serve
                .prefill(&mut source, &prompt(64, 9_000))
                .expect("other prompt");

            // The target held other data; the restore covers the whole saved key.
            serve.reset(&mut target).expect("reset");
            serve
                .prefill(&mut target, &prompt(500, 7_000))
                .expect("dirty");
            serve.reset(&mut target).expect("reset");
            let covered = serve
                .restore_prefix(&mut target, &extended)
                .expect("restore");
            assert_eq!(covered, base.len(), "the whole saved prompt restores");
            assert_eq!(
                target.pos,
                base.len() - 1,
                "the cache position is the entry's"
            );
            serve
                .prefill(&mut target, &extended[covered..])
                .expect("tail prefill");

            serve.reset(&mut cold).expect("reset");
            serve.prefill(&mut cold, &extended).expect("cold prefill");
            assert_eq!(target.pos, cold.pos);
            let last = *extended.last().expect("prompt");
            let restored = next_logits(&mut serve, &target, last);
            let fresh = next_logits(&mut serve, &cold, last);
            assert_eq!(
                restored, fresh,
                "a restored session continues like a cold one"
            );
        }

        /// A different prompt saves over the tip, and the old key no longer restores: the store
        /// holds one entry and answers only for keys the prompt extends.
        #[test]
        #[ignore = "requires a Bolt Q8_0 file and a device"]
        fn the_store_answers_only_for_a_key_the_prompt_extends() {
            let mut serve = serve_of(bolt::load(&bolt_file()).expect("load"));
            let mut first = serve.new_session().expect("first");
            let first_key = prompt(64, 0);
            let other_key = prompt(64, 300);
            serve.reset(&mut first).expect("reset");
            serve.prefill(&mut first, &first_key).expect("prefill");
            assert!(
                serve
                    .save_prefix(&first, &first_key, SaveKind::Tip)
                    .expect("save")
            );
            serve.reset(&mut first).expect("reset");
            serve
                .prefill(&mut first, &other_key)
                .expect("prefill other");
            assert!(
                serve
                    .save_prefix(&first, &other_key, SaveKind::Tip)
                    .expect("save other")
            );

            let mut second = serve.new_session().expect("second");
            serve.reset(&mut second).expect("reset");
            let stale: Vec<u32> = first_key.iter().copied().chain(500..512).collect();
            assert_eq!(serve.restore_prefix(&mut second, &stale).expect("stale"), 0);
            assert_eq!(second.pos, 0, "a miss leaves the session as reset");
            let fresh: Vec<u32> = other_key.iter().copied().chain(600..612).collect();
            assert_eq!(
                serve.restore_prefix(&mut second, &fresh).expect("fresh"),
                other_key.len()
            );
        }
    }
}

#[cfg(feature = "native-cuda")]
pub(crate) use device::start_bolt_engine;
