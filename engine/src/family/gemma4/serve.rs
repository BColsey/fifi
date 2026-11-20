//! The serving wrap of the resident Bolt executor.
//!
//! One session per engine by default; `max_seqs` above one serves several sessions in one decode
//! window. The token side is the shared `DecodeSession`
//! (`family::session`): it owns the committed tokens, the pending token and
//! the finish rules (cancel, length, end of sequence, context limit). The
//! model side is a `TokenStepper` that feeds prompt rows cache-only and reads
//! the capped logits of a decode step; host sampling draws from that row.
//! Prompt rows run the cache-writing layer prefix only (the shared layers
//! after the prefix write no cache cells and feed nothing but the final norm);
//! the pending token's decode runs the head.

#[cfg(any(feature = "native-cuda", test))]
use std::{fs, path::Path};

#[cfg(any(feature = "native-cuda", test))]
use {super::config, crate::EngError};

/// Reads and validates the model directory's `config.json` against the Bolt
/// descriptor; the server starter uses this to dispatch inside the gemma4
/// family. A directory that is not a Bolt checkpoint fails the check.
#[cfg(any(feature = "native-cuda", test))]
pub(crate) fn bolt_configured(model_dir: &Path) -> Result<(), EngError> {
    let json = fs::read(model_dir.join("config.json"))
        .map_err(|error| EngError::Checkpoint(format!("gemma4 config.json: {error}")))?;
    config::check_bolt_config(&json)
}

#[cfg(feature = "native-cuda")]
fn bad(message: &str) -> EngError {
    EngError::BadRequest(format!("gemma4 serve: {message}"))
}

#[cfg(test)]
mod tests {
    use super::bolt_configured;
    use crate::family::gguf_select::candidates;

    const BOLT_CONFIG: &[u8] = include_bytes!("../../../testdata/gemma4/bolt-config.json");

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

    #[test]
    fn the_config_gate_reads_the_directory_of_a_gguf_file_path() {
        let dir = scratch("file-path");
        std::fs::write(dir.join("config.json"), BOLT_CONFIG).expect("write config");
        std::fs::write(dir.join("bolt-q4.gguf"), b"weights").expect("write weights");
        std::fs::write(dir.join("bolt-q8.gguf"), b"weights-bigger").expect("write weights");
        let (found, files) = candidates(&dir.join("bolt-q4.gguf")).expect("file path");
        assert_eq!(found, dir);
        assert_eq!(files.len(), 1);
        bolt_configured(&found).expect("the Bolt config gate passes beside the file");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn bolt_configured_reads_the_pinned_config_shape() {
        let dir = scratch("config");
        assert!(bolt_configured(&dir).is_err());
        std::fs::write(dir.join("config.json"), BOLT_CONFIG).expect("write");
        assert!(bolt_configured(&dir).is_ok());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_bolt_directory_passes_the_config_gate_and_finds_its_gguf() {
        // Checks the two helpers start_gemma4 dispatches on: bolt_configured
        // accepts the pinned Bolt config, and the candidates are the single weights
        // file. The server's engine start itself needs a device and is not run here.
        let dir = scratch("dispatch");
        std::fs::write(dir.join("config.json"), BOLT_CONFIG).expect("write config");
        std::fs::write(dir.join("bolt-e4b-q8_0.gguf"), b"weights").expect("write weights");
        bolt_configured(&dir).expect("the Bolt config gate passes");
        let (_, files) = candidates(&dir).expect("the weights file is found");
        assert_eq!(files[0].path, dir.join("bolt-e4b-q8_0.gguf"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[cfg(feature = "native-cuda")]
mod device {
    use std::path::Path;

    use super::super::budget;
    use super::super::config::MaxSeqs;
    use super::super::resident::bolt::{self, Bolt, BoltSeq, CELLS, VOCAB};
    use crate::{
        EngError,
        family::{
            executor::{ExecutorOps, GenerationState},
            flash_next::factory::StartedEngine,
            seq_plan::{SeqInputs, plan_seqs},
            session::{
                DecodeSession, Limits, PrefixIdentity, PrefixStore, SaveKind, StoreLimits,
                TokenStepper,
            },
            window::{self, WindowRows},
        },
        sched::worker::{ReadyNote, spawn_engine_noted},
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
        /// The cell generation: advances at the attempt of every feed from
        /// position 0, so entries saved by a prompt that was later replaced can
        /// never match again. Mirrors the Flash driver's arena generation
        /// (`flash_next/cuda/driver/chunk.rs`).
        generation: u64,
        /// One-tip prefix store (no roots, no periodic points), keyed by the
        /// cell generation. The snapshot is the model cache position alone: the
        /// KV8 planes are full-length planes whose cells are read only below
        /// the position word, so a rewind re-feeds and overwrites exactly the
        /// cells that the decode past the save point wrote.
        prefix: PrefixStore<usize, (u64, PrefixIdentity)>,
        /// The load-time identity half of the prefix path: constant for one loaded Bolt, so
        /// live behaviour is unchanged; a persisted or cross-step store refuses entries another
        /// load wrote. Empty for now: the Bolt loader reads the GGUF header digest at open
        /// but does not retain it, and threading it means widening `bolt::load`'s return; this
        /// seam takes it when that return widens.
        identity: PrefixIdentity,
        /// The sessions the engine serves at once; the worker reads it after start.
        max_seqs: usize,
        /// The prompt rows the worker feeds per tick while sessions share the model.
        prompt_rows: usize,
        /// False once several sessions share the model: the position-only snapshot would
        /// rewind to cells that another session's decode no longer guarantees, so the store
        /// saves and restores nothing.
        prefix_enabled: bool,
    }

    /// The store's explicit one-tip shape (no roots, no periodic points),
    /// shared by the executor, the start wiring and the tests.
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
        generation: &'a mut u64,
    }

    impl Step<'_> {
        /// The generation advances at the ATTEMPT of a fresh sequence's first
        /// row, not only on success: a failed batch may still have queued the
        /// cell writes, and a stale entry must never survive that.
        fn note_fresh(&mut self) {
            if *self.pos == 0 {
                *self.generation += 1;
            }
        }
    }

    impl TokenStepper for Step<'_> {
        const OPERATION: &'static str = "gemma4_bolt";

        fn feed(&mut self, token: u32) -> Result<(), EngError> {
            self.note_fresh();
            self.model
                .prefill_batch(self.seq, &[token], *self.pos, false)?;
            *self.pos += 1;
            Ok(())
        }

        fn feed_rows(&mut self, tokens: &[u32]) -> Result<(), EngError> {
            for chunk in tokens.chunks(Bolt::PREFILL_BATCH) {
                self.note_fresh();
                self.model
                    .prefill_batch(self.seq, chunk, *self.pos, false)?;
                *self.pos += chunk.len();
            }
            Ok(())
        }

        fn logits(&mut self, token: u32) -> Result<Vec<f32>, EngError> {
            self.note_fresh();
            self.model.decode(self.seq, token, *self.pos)?;
            *self.pos += 1;
            self.model.capped_logits()
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

        fn prompt_rows_per_tick(&self) -> Option<usize> {
            (self.max_seqs > 1).then_some(self.prompt_rows)
        }

        fn new_session(&mut self) -> Result<Self::Session, EngError> {
            Ok(BoltSession {
                core: DecodeSession::default(),
                pos: 0,
                seq: self.model.new_seq()?,
            })
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
            st.core.reset(&mut Step {
                model: &mut self.model,
                seq: &st.seq,
                pos: &mut st.pos,
                generation: &mut self.generation,
            })
        }

        fn check_prompt(&self, prompt_tokens: usize) -> Result<(), EngError> {
            crate::family::session::check_prompt(prompt_tokens, CELLS)
        }

        /// The store's explicit one-tip shape, so the worker's save-point
        /// planning sees what the store really keeps.
        fn prefix_limits(&self) -> StoreLimits {
            ONE_TIP
        }

        fn prefill(&mut self, st: &mut Self::Session, toks: &[u32]) -> Result<(), EngError> {
            st.core.prefill(
                &mut Step {
                    model: &mut self.model,
                    seq: &st.seq,
                    pos: &mut st.pos,
                    generation: &mut self.generation,
                },
                toks,
                CELLS,
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
                context_limit: CELLS,
            };
            st.core.decode(
                &mut Step {
                    model: &mut self.model,
                    seq: &st.seq,
                    pos: &mut st.pos,
                    generation: &mut self.generation,
                },
                generation,
                &limits,
            )
        }

        fn boundary(&mut self, _st: &mut Self::Session) {}

        /// Saves the cache position of the held prompt under the generation
        /// that wrote its cells. The session rules (the pending token, the
        /// strict key match) are `DecodeSession`'s; the model state at the save
        /// point is the position word alone.
        fn save_prefix(
            &mut self,
            st: &Self::Session,
            key: &[u32],
            kind: SaveKind,
        ) -> Result<bool, EngError> {
            if !self.prefix_enabled {
                return Ok(false);
            }
            let path = (self.generation, self.identity.clone());
            let fed = st.pos;
            st.core
                .save_prefix(&mut self.prefix, key, path, fed, kind, || Ok(st.pos))
        }

        /// Rewinds the cache position to the saved entry of the current
        /// generation. The KV8 cells at and above it hold the decode that
        /// followed the save; a re-feed overwrites them, so the restored state
        /// is exact. A different prompt fed in between advanced the generation,
        /// so its entry never matches — the stale-entry guard. The tail prefill
        /// that follows a restore overwrites every cell above the boundary, so
        /// every stored entry with a longer key is dropped before any row is
        /// written (`drop_above`, as the Flash executor does).
        fn restore_prefix(
            &mut self,
            st: &mut Self::Session,
            prompt: &[u32],
        ) -> Result<usize, EngError> {
            if !self.prefix_enabled {
                return Ok(0);
            }
            let path = (self.generation, self.identity.clone());
            let BoltSession { core, pos, .. } = st;
            let restored = core.restore_prefix(&mut self.prefix, prompt, path, |saved| {
                *pos = *saved;
                Ok(())
            })?;
            if restored > 0 {
                self.prefix.drop_above(restored);
            }
            Ok(restored)
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
            (
                core,
                Step {
                    model: &mut self.model,
                    seq,
                    pos,
                    generation: &mut self.generation,
                },
                Limits {
                    eos: &self.eos,
                    sampler_vocab: VOCAB,
                    context_limit: CELLS,
                },
            )
        }

        fn check_row(&self, session: &BoltSession, token: u32) -> Result<(), EngError> {
            if session.pos >= CELLS {
                return Err(EngError::ContextLimit(format!(
                    "position {} reaches the {CELLS} cell context",
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

        /// One token of every session through one step. A session that feeds from position 0
        /// advances the cell generation at the attempt, as a solo feed does.
        fn step_rows(
            &mut self,
            sessions: &mut [&mut BoltSession],
            tokens: &[u32],
        ) -> Result<Vec<Vec<f32>>, EngError> {
            if sessions.iter().any(|session| session.pos == 0) {
                self.generation += 1;
            }
            let seqs: Vec<&BoltSeq> = sessions.iter().map(|session| &session.seq).collect();
            let positions: Vec<usize> = sessions.iter().map(|session| session.pos).collect();
            let rows = self.model.decode_rows(&seqs, tokens, &positions)?;
            for session in sessions.iter_mut() {
                session.pos += 1;
            }
            Ok(rows)
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
            context: CELLS as u64,
            row_cap: Bolt::PREFILL_BATCH,
            ceiling: Bolt::PREFILL_BATCH,
        };
        plan_seqs(
            inputs,
            match seqs {
                MaxSeqs::Count(count) => Some(count),
                MaxSeqs::Auto => None,
            },
        )
    }

    /// `model_path` is the model directory or one `.gguf` file in it. A directory with several
    /// `.gguf` files serves the highest-precision one that the device memory admits; `seqs`
    /// is the explicit session count or the count the free memory funds. Several sessions turn
    /// the position-only prefix snapshot off and the decode graph off, and the start line says so.
    pub(crate) fn start_bolt_engine(
        model_path: &Path,
        tokenizer_dir: &Path,
        ready_note: ReadyNote,
        seqs: MaxSeqs,
        prompt_rows: Option<usize>,
    ) -> Result<StartedEngine, EngError> {
        let (_, files) = crate::family::gguf_select::candidates(model_path)?;
        let bundle = crate::tokenizer::gemma::load(tokenizer_dir)?;
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
        let engine = spawn_engine_noted(Bolt::PREFILL_BATCH, ready_note, move |_| {
            let mut model = bolt::load_choosing(&files, None)?;
            let first = model.new_seq()?;
            let plan = plan_sessions(&model, seqs)?;
            eprintln!("gemma4 bolt: {}", plan.describe());
            if plan.seqs > 1 {
                model.reserve_window(plan.seqs)?;
                model.use_decode_graph(false);
                eprintln!(
                    "gemma4 bolt: {} sessions share the model, so the position-only prefix snapshot and the decode graph are off",
                    plan.seqs
                );
            }
            let serve = BoltServe {
                model,
                eos: eos.clone(),
                generation: 0,
                prefix: PrefixStore::with_limits(ONE_TIP),
                identity: PrefixIdentity::default(),
                max_seqs: plan.seqs,
                prompt_rows: prompt_rows.unwrap_or(budget::BOLT_AUTO_PROMPT_ROWS),
                prefix_enabled: plan.seqs == 1,
            };
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
        Ok(StartedEngine {
            engine,
            bundle,
            context_limit: u32::try_from(CELLS).expect("cells fit u32"),
        })
    }

    #[cfg(test)]
    mod prefix_tests {
        use super::*;

        /// Panics when the pinned Bolt file is not configured: an explicit
        /// `--ignored` run without `BOLT_Q8_FILE` is a setup error, not a pass.
        fn pinned_bolt() -> std::path::PathBuf {
            std::env::var("BOLT_Q8_FILE")
                .map(std::path::PathBuf::from)
                .expect("BOLT_Q8_FILE must point at the pinned Bolt q8 file")
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
        #[ignore = "requires the pinned Bolt file and a device"]
        fn a_restore_past_a_decode_matches_the_cold_replay() {
            let mut model: Bolt = bolt::load(&pinned_bolt(), None).expect("load");
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
            // tokens (one PREFILL_BATCH of rows), then draw greedily.
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
        #[ignore = "requires the pinned Bolt file and a device"]
        fn a_graphed_greedy_run_matches_the_eager_run() {
            let mut model: Bolt = bolt::load(&pinned_bolt(), None).expect("load");
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
        #[ignore = "requires the pinned Bolt file and a device"]
        fn a_graphed_split_run_matches_the_eager_split_run() {
            let mut model: Bolt = bolt::load(&pinned_bolt(), None).expect("load");
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

        /// A one-token prompt feeds nothing in prefill, so its first decode
        /// writes cell 0 through `logits`: that decode must advance the cell
        /// generation, or a stale entry would match over the overwritten cell.
        #[test]
        #[ignore = "requires the pinned Bolt file and a device"]
        fn a_one_token_prompt_advances_the_generation_before_its_first_decode() {
            let mut serve = BoltServe {
                model: bolt::load(&pinned_bolt(), None).expect("load"),
                eos: vec![0],
                generation: 0,
                prefix: PrefixStore::with_limits(ONE_TIP),
                identity: PrefixIdentity::default(),
                max_seqs: 1,
                prompt_rows: budget::BOLT_AUTO_PROMPT_ROWS,
                prefix_enabled: true,
            };
            let mut st = serve.new_session().expect("session");
            let first: Vec<u32> = (0..64).map(|index| 100 + (index % 977)).collect();
            let first_ext: Vec<u32> = first.iter().copied().chain(500..512).collect();

            serve.reset(&mut st).expect("reset");
            serve.prefill(&mut st, &first).expect("prefill first");
            assert!(serve.save_prefix(&st, &first, SaveKind::Tip).expect("save"));

            // The one-token prompt: prefill holds its only token pending, so
            // nothing is fed and the position stays 0; the decode step then
            // writes cell 0.
            serve.reset(&mut st).expect("reset");
            serve.prefill(&mut st, &[42]).expect("prefill one token");
            assert_eq!(st.pos, 0, "the pending token is not fed yet");
            let mut step = Step {
                model: &mut serve.model,
                seq: &st.seq,
                pos: &mut st.pos,
                generation: &mut serve.generation,
            };
            use crate::family::session::TokenStepper as _;
            step.logits(42).expect("first decode");
            assert!(
                serve.generation >= 2,
                "the first decode at position 0 must advance the generation"
            );

            // The stale entry can no longer match the overwritten cell 0.
            let covered = serve
                .restore_prefix(&mut st, &first_ext)
                .expect("stale restore");
            assert_eq!(covered, 0, "the stale entry must cover nothing");
        }

        /// The seam refuses a stale entry: after a different prompt's prefill
        /// advanced the cell generation, restoring onto the first prompt's
        /// extension covers nothing and leaves the session unchanged; the
        /// current prompt's own entry still restores and rewinds.
        #[test]
        #[ignore = "requires the pinned Bolt file and a device"]
        fn the_seam_refuses_a_stale_entry_and_restores_the_current_one() {
            let mut serve = BoltServe {
                model: bolt::load(&pinned_bolt(), None).expect("load"),
                eos: vec![0],
                generation: 0,
                prefix: PrefixStore::with_limits(ONE_TIP),
                identity: PrefixIdentity::default(),
                max_seqs: 1,
                prompt_rows: budget::BOLT_AUTO_PROMPT_ROWS,
                prefix_enabled: true,
            };
            let mut st = serve.new_session().expect("session");
            let n = 64;
            let first: Vec<u32> = (0..n as u32).map(|index| 100 + (index % 977)).collect();
            let other: Vec<u32> = (0..n as u32).map(|index| 200 + (index % 811)).collect();
            let first_ext: Vec<u32> = first.iter().copied().chain(500..512).collect();
            let other_ext: Vec<u32> = other.iter().copied().chain(600..612).collect();

            serve.reset(&mut st).expect("reset");
            serve.prefill(&mut st, &first).expect("prefill first");
            assert!(serve.save_prefix(&st, &first, SaveKind::Tip).expect("save"));

            // A different prompt feeds from position 0: the generation
            // advances and the first prompt's entry can never match again.
            serve.reset(&mut st).expect("reset");
            serve.prefill(&mut st, &other).expect("prefill other");
            let covered = serve
                .restore_prefix(&mut st, &first_ext)
                .expect("stale restore");
            assert_eq!(covered, 0, "the stale entry must cover nothing");
            assert_eq!(
                st.core.history(),
                &other[..],
                "the session must be unchanged"
            );

            // The current generation's entry still restores and rewinds.
            assert!(serve.save_prefix(&st, &other, SaveKind::Tip).expect("save"));
            serve.reset(&mut st).expect("reset");
            let covered = serve
                .restore_prefix(&mut st, &other_ext)
                .expect("fresh restore");
            assert_eq!(covered, n, "the current entry must cover the prompt");
            assert_eq!(
                st.pos,
                n - 1,
                "the cache position must rewind to the boundary"
            );
            serve
                .prefill(&mut st, &other_ext[n..])
                .expect("tail prefill");
            assert_eq!(
                st.core.history(),
                &other_ext[..],
                "the tail must continue the sequence"
            );
        }
    }
}

#[cfg(feature = "native-cuda")]
pub(crate) use device::start_bolt_engine;
