//! The host feeds of the Flash-Next CUDA step: batched n-gram row reads and the token-start write.
//!
//! WHAT:
//! - [`RowReader`] reads and decodes global n-gram rows for a token (16 rows) or a prefill chunk
//!   (16 per token). It sorts and de-duplicates the row ids, splits the sorted run over a pool of
//!   reader threads, and each thread calls the existing [`NgramFile::read_row`] (a buffered
//!   positioned read) and [`NgramDecoder::decode_row`]. [`RowReader::submit`] returns a
//!   [`PendingRows`] at once so the caller can overlap the reads with device work and
//!   [`PendingRows::wait`] for the [`RowBlock`] later.
//! - [`plan_chunk`] hashes a chunk of tokens with the existing `ple::plan_chunk_rows` and returns
//!   the global row ids of each token; [`global_rows`] maps one `RowPlan` of `ple::plan_rows` the
//!   same way.
//! - [`write_start`] validates one token start and hands it to a [`StartSink`] in the order of the
//!   Qwen3.6 `write_start` (`qwen36/cuda/step.rs`): the step parameters (token and position), then
//!   the initial hyper stream and the PLE embedding. The device sink is in `device`.
//! - `device` (native CUDA builds) runs the PLE `key_proj` and `value_proj` on the card through
//!   `CudaContext::dense_gemv` and writes the start block to the device words.
//!
//! WHY a pool and a sorted run: the table is 30.4 GiB on local NVMe and a row is 102 bytes that
//! cost one 4 KiB page. Measured on the real file (`doc/flash-next-ngram-latency.md`), a buffered
//! read costs about 52 us cold and 1 us warm per row at queue depth 1, and a 16-row token costs
//! 837 us cold on one thread against 160-200 us p50 (p99 238-323 us) through this reader on 32
//! threads. A prefill chunk of 4096 tokens (65,528 distinct rows) costs 242-244 ms through
//! `RowReader::read` plus `embedding()` for every token; the raw read without decode took 142 ms on 32 threads in the bench.
//! One thread would need seconds. Sorting by id sorts by file offset, so each thread reads a
//! contiguous run. The pool lives as long as the reader, so a decode token pays a channel
//! hand-off and no thread spawn.
//!
//! Hashing and the history stay in `ple.rs`; this file adds no second hash, row planner or decoder.
//! No user-space row cache: the page cache holds the rows, and a cache is justified only by a
//! measured low hit rate.

// Staged: the Flash-Next CUDA step is the first caller.
#![cfg_attr(not(test), allow(dead_code))]

use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
        mpsc,
    },
    thread,
};

use crate::{
    EngError,
    family::flash_next::ple::{self, PleShape, PleWeights},
    weights::exl3::{
        ngram::{NgramDecoder, ROW_DIM},
        ngram_file::NgramFile,
    },
};

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(in crate::family) mod device;

/// Reader threads when the caller does not choose. They wait on the drive, not on the CPU, so the
/// count may exceed the cores. A 4096-token chunk was read faster on 32 threads than on 16 (142
/// against 226 ms, read only); more than 32 was not measured.
pub(in crate::family) const DEFAULT_THREADS: usize = 32;
const MAX_THREADS: usize = 128;

/// Page cache the n-gram table needs reserved in the host budget (`HostBudget::page_cache` of
/// `tier_plan`): none. Measured on the real file (`doc/flash-next-ngram-latency.md`), a decode
/// token with all 16 rows cold costs 160-200 us p50 and 238-323 us p99 through [`RowReader`] on 32
/// threads (three runs, 400 tokens each, machine under load), about 1% of a token, and the reads can
/// start before layer 0; a prefill chunk reads one chunk ahead of its
/// compute. A warm page is a bonus, not a requirement, so the pinned expert store may take the RAM
/// and the kernel keeps whatever rows fit. A reserve is justified only by a measured low hit rate
/// on real transcripts (the trace study M-N2 was not run).
pub(in crate::family) const NGRAM_PAGE_CACHE_RESERVE_BYTES: u64 = 0;

/// Host memory a read of `rows` distinct rows holds while in flight: the decoded F32 block and its
/// ids, plus the per-thread copies of the sorted runs. 4096 tokens at 16 rows is 65,536 rows and
/// 42 MB of values; a double-buffered chunk worker holds two such blocks.
pub(in crate::family) fn ngram_staging_bytes(rows: u64) -> u64 {
    rows * (ROW_DIM as u64 * 4 + 8 + 8)
}

/// The decoded rows of a read: the sorted distinct ids and `ROW_DIM` values for each.
#[derive(Debug, Clone, PartialEq)]
pub(in crate::family) struct RowBlock {
    ids: Vec<u64>,
    values: Vec<f32>,
}

impl RowBlock {
    /// Number of distinct rows.
    pub(in crate::family) fn len(&self) -> usize {
        self.ids.len()
    }

    /// The distinct row ids, ascending.
    pub(in crate::family) fn ids(&self) -> &[u64] {
        &self.ids
    }

    /// The decoded values of global row `row`, if the read included it.
    pub(in crate::family) fn get(&self, row: u64) -> Option<&[f32]> {
        let at = self.ids.binary_search(&row).ok()?;
        Some(&self.values[at * ROW_DIM..(at + 1) * ROW_DIM])
    }

    /// The PLE embedding of one token: its rows in head order, each value times `scale`
    /// (`weight_scale` of `ple::step`). A row the read did not include is `BadRequest`.
    pub(in crate::family) fn embedding(
        &self,
        rows: &[u64],
        scale: f32,
    ) -> Result<Vec<f32>, EngError> {
        let mut out = Vec::with_capacity(rows.len() * ROW_DIM);
        for &row in rows {
            let values = self
                .get(row)
                .ok_or_else(|| EngError::BadRequest(format!("n-gram row {row} was not read")))?;
            out.extend(values.iter().map(|v| v * scale));
        }
        Ok(out)
    }
}

struct Job {
    rows: Vec<u64>,
    part: usize,
    reply: mpsc::Sender<(usize, Result<Vec<f32>, EngError>)>,
}

/// A submitted read. [`Self::wait`] blocks until every part is back.
pub(in crate::family) struct PendingRows {
    ids: Vec<u64>,
    parts: usize,
    replies: mpsc::Receiver<(usize, Result<Vec<f32>, EngError>)>,
}

impl PendingRows {
    pub(in crate::family) fn wait(self) -> Result<RowBlock, EngError> {
        let mut parts: Vec<Option<Vec<f32>>> = vec![None; self.parts];
        for _ in 0..self.parts {
            let (part, result) = self.replies.recv().map_err(|_| {
                EngError::Checkpoint("an n-gram reader thread ended before it replied".to_owned())
            })?;
            parts[part] = Some(result?);
        }
        let mut values = Vec::with_capacity(self.ids.len() * ROW_DIM);
        for part in parts {
            values.extend(part.ok_or_else(|| {
                EngError::Checkpoint("an n-gram reader part is missing".to_owned())
            })?);
        }
        if values.len() != self.ids.len() * ROW_DIM {
            return Err(EngError::Checkpoint(format!(
                "an n-gram read returned {} values for {} rows",
                values.len(),
                self.ids.len()
            )));
        }
        Ok(RowBlock {
            ids: self.ids,
            values,
        })
    }
}

type ReadRun = dyn Fn(&[u64]) -> Result<Vec<f32>, EngError> + Send + Sync;

/// Batched reads of the n-gram table through a pool of reader threads.
///
/// Parts of one read go to consecutive workers starting at a rotating counter, so reads in flight
/// together (a decode token behind a chunk, a double-buffered chunk worker) spread over the pool
/// instead of queuing on the same first workers.
///
/// A panic in a worker ends that worker. Reads that had a part on it fail with a `Checkpoint`
/// error from [`PendingRows::wait`], and any later [`Self::submit`] that puts a part on it fails
/// with a `Checkpoint` error: the pool does not respawn, so the reader is unusable and must be
/// rebuilt. Dropping the reader closes the queues; the workers finish the jobs already queued and
/// exit, so a [`PendingRows`] still waits to a complete block after the reader is dropped.
pub(in crate::family) struct RowReader {
    workers: Vec<mpsc::Sender<Job>>,
    next: AtomicUsize,
}

impl RowReader {
    /// `threads` is clamped to `1..=128`. The file was opened through the local-storage guard by
    /// `FlashPackFiles`; the reader only reads it.
    pub(in crate::family) fn new(
        file: Arc<NgramFile>,
        decoder: Arc<NgramDecoder>,
        threads: usize,
    ) -> Result<Self, EngError> {
        Self::with_reader(
            Arc::new(move |rows: &[u64]| read_run(&file, &decoder, rows)),
            threads,
        )
    }

    fn with_reader(read: Arc<ReadRun>, threads: usize) -> Result<Self, EngError> {
        let threads = threads.clamp(1, MAX_THREADS);
        let mut workers = Vec::with_capacity(threads);
        for index in 0..threads {
            let (sender, jobs) = mpsc::channel::<Job>();
            let read = Arc::clone(&read);
            thread::Builder::new()
                .name(format!("flash-ngram-read-{index}"))
                .spawn(move || {
                    for job in jobs {
                        let result = read(&job.rows);
                        // The caller dropped its PendingRows: nobody wants the rows.
                        let _ = job.reply.send((job.part, result));
                    }
                })
                .map_err(|error| EngError::MemoryPlan(format!("n-gram reader thread: {error}")))?;
            workers.push(sender);
        }
        Ok(Self {
            workers,
            next: AtomicUsize::new(0),
        })
    }

    /// Queues the read of `rows` (any order, repeats allowed) and returns at once. The distinct
    /// rows are split into `min(rows, threads)` consecutive runs whose lengths differ by at most
    /// one, so every thread of the pool gets work when there are rows for it.
    pub(in crate::family) fn submit(&self, rows: &[u64]) -> Result<PendingRows, EngError> {
        let mut ids = rows.to_vec();
        ids.sort_unstable();
        ids.dedup();
        let (reply, replies) = mpsc::channel();
        let parts = ids.len().min(self.workers.len());
        if let Some(size) = ids.len().checked_div(parts) {
            let extra = ids.len() % parts;
            let first = self.next.fetch_add(parts, Ordering::Relaxed);
            let mut at = 0;
            for part in 0..parts {
                let len = size + usize::from(part < extra);
                self.workers[(first + part) % self.workers.len()]
                    .send(Job {
                        rows: ids[at..at + len].to_vec(),
                        part,
                        reply: reply.clone(),
                    })
                    .map_err(|_| {
                        EngError::Checkpoint("an n-gram reader thread has ended".to_owned())
                    })?;
                at += len;
            }
        }
        Ok(PendingRows {
            ids,
            parts,
            replies,
        })
    }

    /// [`Self::submit`] and [`PendingRows::wait`].
    pub(in crate::family) fn read(&self, rows: &[u64]) -> Result<RowBlock, EngError> {
        self.submit(rows)?.wait()
    }
}

/// Reads and decodes a run of ascending rows on the calling thread.
fn read_run(file: &NgramFile, decoder: &NgramDecoder, rows: &[u64]) -> Result<Vec<f32>, EngError> {
    let mut packed = vec![0u8; file.table().row_bytes()];
    let mut out = Vec::with_capacity(rows.len() * ROW_DIM);
    let mut values = [0f32; ROW_DIM];
    for &row in rows {
        file.read_row(row, &mut packed)?;
        decoder.decode_row(row, &packed, &mut values)?;
        out.extend_from_slice(&values);
    }
    Ok(out)
}

/// The first global row of each table, then the total: `starts[t] + local` is the global row.
fn table_starts(shard_rows: &[u64]) -> Result<Vec<u64>, EngError> {
    let mut starts = Vec::with_capacity(shard_rows.len());
    let mut sum = 0u64;
    for &count in shard_rows {
        starts.push(sum);
        sum = sum
            .checked_add(count)
            .ok_or_else(|| EngError::Checkpoint("ple table rows overflow".to_owned()))?;
    }
    Ok(starts)
}

/// Global row ids of the `(table, local)` pairs of a `RowPlan`, in head order.
pub(in crate::family) fn global_rows(
    rows: &[(usize, u32)],
    shard_rows: &[u64],
) -> Result<Vec<u64>, EngError> {
    let starts = table_starts(shard_rows)?;
    rows.iter()
        .map(|&(table, local)| {
            starts
                .get(table)
                .map(|start| start + u64::from(local))
                .ok_or_else(|| EngError::Checkpoint("ple row table is missing".to_owned()))
        })
        .collect()
}

/// The rows of a chunk of tokens: per token in head order, and the history after the chunk.
#[derive(Debug, PartialEq, Eq)]
pub(in crate::family) struct ChunkRows {
    pub(in crate::family) per_token: Vec<Vec<u64>>,
    pub(in crate::family) history: Vec<u32>,
}

impl ChunkRows {
    /// Every row of the chunk, repeats included; [`RowReader::submit`] sorts and de-duplicates.
    pub(in crate::family) fn all(&self) -> Vec<u64> {
        self.per_token.concat()
    }
}

/// Plans a whole chunk up front, so one sorted read serves it (`ple::plan_chunk_rows`).
pub(in crate::family) fn plan_chunk<W>(
    shape: &PleShape,
    weights: &PleWeights<'_, W>,
    history: &[u32],
    tokens: &[u32],
) -> Result<ChunkRows, EngError> {
    let plan = ple::plan_chunk_rows(shape, weights, history, tokens)?;
    let per_token = plan
        .rows
        .iter()
        .map(|rows| global_rows(rows, weights.shard_rows))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(ChunkRows {
        per_token,
        history: plan.history,
    })
}

/// Where a token start lands. The device sink writes the words block and two buffers of the
/// Flash state, ordered on the step stream; a test sink records.
pub(in crate::family) trait StartSink {
    /// The step parameters: the token to embed and its position. The sink builds and validates the
    /// block (`StepParams::new`: a negative token or an out-of-range position is refused).
    fn write_step_params(&mut self, token: i32, pos: i32) -> Result<(), EngError>;
    /// The initial hyper stream, `branches * hidden` values.
    fn write_stream(&mut self, stream: &[f32]) -> Result<(), EngError>;
    /// The PLE embedding, `ple_embed` values, the input of the key and value projections.
    fn write_ple_embedding(&mut self, embedding: &[f32]) -> Result<(), EngError>;
}

/// The sizes a token start is checked against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::family) struct StartShape {
    /// The sampler vocabulary, the largest id the argmax can draw plus one. It is NOT the width of
    /// the embedding table: the table has padding rows above it that would embed and run, so ids
    /// from here up to the table width are refused like the Qwen3.6 `check_token_id`.
    pub(in crate::family) sampler_vocab: usize,
    /// `branches * hidden`.
    pub(in crate::family) stream: usize,
    /// `ple_embed`.
    pub(in crate::family) ple_embed: usize,
}

/// Writes the start block of the token at position `pos`.
///
/// Every check comes before the first write, so a refused token changes nothing (the rule of the
/// Qwen3.6 `write_start`): the token is inside the sampler vocabulary, both vectors have their
/// width, and the token and position fit an `i32`.
pub(in crate::family) fn write_start<S: StartSink>(
    sink: &mut S,
    shape: &StartShape,
    token: u32,
    pos: usize,
    stream: &[f32],
    ple_embedding: &[f32],
) -> Result<(), EngError> {
    if usize::try_from(token).is_ok_and(|id| id >= shape.sampler_vocab) {
        return Err(EngError::BadRequest(format!(
            "token id {token} is outside the sampler vocabulary 0..{}",
            shape.sampler_vocab
        )));
    }
    if stream.len() != shape.stream {
        return Err(EngError::BadRequest(format!(
            "the start stream has {} values, expected {}",
            stream.len(),
            shape.stream
        )));
    }
    if ple_embedding.len() != shape.ple_embed {
        return Err(EngError::BadRequest(format!(
            "the PLE embedding has {} values, expected {}",
            ple_embedding.len(),
            shape.ple_embed
        )));
    }
    let too_big = |what| EngError::BadRequest(format!("step: the {what} does not fit an i32"));
    let token = i32::try_from(token).map_err(|_| too_big("token"))?;
    let pos = i32::try_from(pos).map_err(|_| too_big("position"))?;
    sink.write_step_params(token, pos)?;
    sink.write_stream(stream)?;
    sink.write_ple_embedding(ple_embedding)
}

#[cfg(test)]
mod tests;
