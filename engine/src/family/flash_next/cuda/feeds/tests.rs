//! The feeds against the committed real rows of the Flash-Next n-gram table.
//!
//! - Rows: the sparse n-gram file of `tests/fixtures/flash_next_exl3_ngram` carries the real
//!   header, the auxiliary tensors and the 80 real rows of a five-token prompt (16 per token).
//!   Hashing the prompt through `ple::plan_rows` / `plan_chunk_rows` must land on the recorded
//!   global rows, and the pool read must return the recorded decoded bits, for any thread count,
//!   any request order and with repeats.
//! - Start block: a recording sink shows the write order and that a refusal writes nothing.
//! - Real pack (ignored): `FIFI_FLASH_NEXT_EXL3_PACK` names the full local pack. The same prompt
//!   is hashed and read from the 30.4 GiB table (the bits equal the fixture's, same revision), a
//!   chunk is read sorted, and the per-token latency of the production reader is printed cold and
//!   warm.

use std::{
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use super::{
    DEFAULT_THREADS, NGRAM_PAGE_CACHE_RESERVE_BYTES, RowReader, StartShape, StartSink, global_rows,
    ngram_staging_bytes, plan_chunk, write_start,
};
use crate::{
    EngError,
    family::flash_next::ple::{self, PleDims, PleShape, PleWeights},
    storage_guard::LocalGuard,
    weights::exl3::{
        ngram::{NgramAux, NgramDecoder, ROW_DIM},
        ngram_file::{
            NgramFile,
            tests::{DECODED, KEY, LAYOUT, manifest_integers, write_table},
        },
    },
};

const PROMPT: [u32; 5] = [760, 6511, 314, 9338, 369];
const EOS: u32 = 248_044;

fn dims() -> PleDims {
    PleDims {
        branches: 4,
        hidden: 2560,
        ngram: 3,
        heads_per_ngram: 8,
        head_dim: ROW_DIM as u32,
        conv_kernel: 4,
        eos: EOS,
        eps: 1e-6,
    }
}

fn le_i64(bytes: &[u8]) -> Vec<i64> {
    bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|word| i64::from_le_bytes(*word))
        .collect()
}

/// The hash tensors of a table file, as the feed reads them.
struct Hash {
    multipliers: Vec<i64>,
    vocab: Vec<i64>,
    offsets: Vec<i64>,
    shard_rows: [u64; 1],
}

impl Hash {
    fn of(file: &NgramFile) -> Self {
        Self {
            multipliers: le_i64(&file.read_aux(NgramAux::LayerMultipliers).unwrap()),
            vocab: le_i64(&file.read_aux(NgramAux::HeadVocabSizes).unwrap()),
            offsets: le_i64(&file.read_aux(NgramAux::HeadOffsets).unwrap()),
            shard_rows: [file.table().rows()],
        }
    }

    /// The planner's view: only the hash tensors and the one virtual table are read.
    fn weights<'a>(&'a self, unit: &'a [(); 1], key: &'a ()) -> PleWeights<'a, ()> {
        PleWeights {
            multipliers: &self.multipliers,
            vocab: &self.vocab,
            offsets: &self.offsets,
            shards: unit,
            shard_rows: &self.shard_rows,
            weight_scale: 1.0,
            key,
            value: key,
            norm_key: &[],
            norm_query: &[],
            norm_conv: &[],
            conv: &[],
        }
    }
}

fn decoder_of(file: &NgramFile) -> NgramDecoder {
    let bias: Vec<u16> = file
        .read_aux(NgramAux::HeadBias)
        .unwrap()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u16::from_le_bytes(*pair))
        .collect();
    NgramDecoder::new(
        file.table(),
        &le_i64(&file.read_aux(NgramAux::HeadOffsets).unwrap()),
        &le_i64(&file.read_aux(NgramAux::HeadVocabSizes).unwrap()),
        &bias,
    )
    .unwrap()
}

fn fixture_file(dir: &Path) -> Arc<NgramFile> {
    let path = dir.join("ngram_embedding.safetensors");
    write_table(&path);
    Arc::new(NgramFile::open(&path, &LocalGuard::new().unwrap(), KEY, LAYOUT).unwrap())
}

fn recorded_rows() -> Vec<u64> {
    manifest_integers("global_rows")
        .into_iter()
        .map(|row| row as u64)
        .collect()
}

/// The recorded decoded bits of the `index`-th recorded row.
fn recorded_values(index: usize) -> Vec<u32> {
    (0..ROW_DIM)
        .map(|position| {
            let at = (index * ROW_DIM + position) * 4;
            u32::from_le_bytes(DECODED[at..at + 4].try_into().unwrap())
        })
        .collect()
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

fn reader(file: &Arc<NgramFile>, threads: usize) -> RowReader {
    RowReader::new(Arc::clone(file), Arc::new(decoder_of(file)), threads).unwrap()
}

#[test]
fn hashing_the_prompt_lands_on_the_recorded_rows_token_by_token_and_as_a_chunk() {
    let dir = tempfile::tempdir().unwrap();
    let file = fixture_file(dir.path());
    let hash = Hash::of(&file);
    let (unit, key) = ([()], ());
    let weights = hash.weights(&unit, &key);
    let shape = PleShape::new(dims()).unwrap();
    let recorded = recorded_rows();
    assert_eq!(recorded.len(), 80);

    // One token at a time, carrying the history (the decode path).
    let mut history = Vec::new();
    for (index, &token) in PROMPT.iter().enumerate() {
        let plan = ple::plan_rows(&shape, &weights, &history, token).unwrap();
        let rows = global_rows(&plan.rows, &hash.shard_rows).unwrap();
        assert_eq!(
            rows,
            recorded[index * 16..(index + 1) * 16],
            "token {index}"
        );
        history = plan.history;
    }

    // The whole chunk at once (the prefill path), from an empty history.
    let chunk = plan_chunk(&shape, &weights, &[], &PROMPT).unwrap();
    assert_eq!(chunk.all(), recorded);
    assert_eq!(chunk.history, history);
    assert_eq!(chunk.history, vec![9338, 369]);

    // Split in two chunks the history carries over.
    let first = plan_chunk(&shape, &weights, &[], &PROMPT[..2]).unwrap();
    let second = plan_chunk(&shape, &weights, &first.history, &PROMPT[2..]).unwrap();
    assert_eq!([first.all(), second.all()].concat(), recorded);
}

#[test]
fn the_pool_returns_the_recorded_bits_for_any_thread_count_order_and_repeats() {
    let dir = tempfile::tempdir().unwrap();
    let file = fixture_file(dir.path());
    let recorded = recorded_rows();
    // Reversed, every row twice: the block is sorted and distinct whatever the request.
    let request: Vec<u64> = recorded
        .iter()
        .rev()
        .chain(recorded.iter())
        .copied()
        .collect();
    for threads in [1, 2, 3, 7, 32, 500] {
        let reader = reader(&file, threads);
        let block = reader.read(&request).unwrap();
        let mut distinct = recorded.clone();
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(block.ids(), distinct, "{threads} threads");
        assert_eq!(block.len(), distinct.len());
        for (index, &row) in recorded.iter().enumerate() {
            assert_eq!(
                bits(block.get(row).unwrap()),
                recorded_values(index),
                "{threads} threads, row {row}"
            );
        }
        let absent = (recorded[0] + 1..)
            .find(|row| !recorded.contains(row))
            .unwrap();
        assert!(block.get(absent).is_none());
    }
}

#[test]
fn a_token_embedding_is_its_sixteen_rows_in_head_order_times_the_scale() {
    let dir = tempfile::tempdir().unwrap();
    let file = fixture_file(dir.path());
    let hash = Hash::of(&file);
    let (unit, key) = ([()], ());
    let weights = hash.weights(&unit, &key);
    let shape = PleShape::new(dims()).unwrap();
    let chunk = plan_chunk(&shape, &weights, &[], &PROMPT).unwrap();
    let block = reader(&file, DEFAULT_THREADS).read(&chunk.all()).unwrap();
    assert_eq!(block.len(), 80);
    for (token, rows) in chunk.per_token.iter().enumerate() {
        let embedding = block.embedding(rows, 1.0).unwrap();
        assert_eq!(embedding.len(), shape.embed());
        let want: Vec<u32> = (0..16)
            .flat_map(|head| recorded_values(token * 16 + head))
            .collect();
        assert_eq!(bits(&embedding), want, "token {token}");
        let doubled = block.embedding(rows, 2.0).unwrap();
        assert!(doubled.iter().zip(&embedding).all(|(a, b)| *a == b * 2.0));
    }
    assert!(block.embedding(&[chunk.per_token[0][0] + 1], 1.0).is_err());
}

#[test]
fn submissions_overlap_and_an_empty_or_failing_read_is_reported() {
    let dir = tempfile::tempdir().unwrap();
    let file = fixture_file(dir.path());
    let recorded = recorded_rows();
    let reader = reader(&file, 8);
    // Two reads in flight at once, awaited in the other order.
    let a = reader.submit(&recorded[..40]).unwrap();
    let b = reader.submit(&recorded[40..]).unwrap();
    let (b, a) = (b.wait().unwrap(), a.wait().unwrap());
    assert_eq!((a.len(), b.len()), (40, 40));
    assert_eq!(bits(b.get(recorded[79]).unwrap()), recorded_values(79));

    let empty = reader.read(&[]).unwrap();
    assert_eq!(empty.len(), 0);

    // A row in the padding after the last head decodes to no head: BadRequest, not a panic.
    let past_heads = file.table().rows() - 10;
    match reader.read(&[recorded[0], past_heads]) {
        Err(EngError::BadRequest(message)) => assert!(message.contains("hash head"), "{message}"),
        other => panic!("{other:?}"),
    }
    // A row outside the table is refused by the file.
    assert!(reader.read(&[file.table().rows()]).is_err());
    // The pool still works after a failed read.
    assert_eq!(reader.read(&recorded[..16]).unwrap().len(), 16);
}

#[derive(Default)]
struct Recorder {
    events: Vec<String>,
}

impl StartSink for Recorder {
    fn write_step_params(&mut self, token: i32, pos: i32) -> Result<(), EngError> {
        self.events.push(format!("params {token} {pos}"));
        Ok(())
    }
    fn write_stream(&mut self, stream: &[f32]) -> Result<(), EngError> {
        self.events.push(format!("stream {}", stream.len()));
        Ok(())
    }
    fn write_ple_embedding(&mut self, embedding: &[f32]) -> Result<(), EngError> {
        self.events.push(format!("embedding {}", embedding.len()));
        Ok(())
    }
}

/// A sampler vocabulary of 248,070 inside an embedding table of 248,320 rows: ids 248,070 to
/// 248,319 are padding rows that would embed and run but that the argmax can never draw.
const SHAPE: StartShape = StartShape {
    sampler_vocab: 248_070,
    stream: 8,
    ple_embed: 4,
};

#[test]
fn a_token_start_writes_the_params_then_the_stream_then_the_embedding() {
    let mut sink = Recorder::default();
    write_start(&mut sink, &SHAPE, 760, 12, &[0.0; 8], &[0.0; 4]).unwrap();
    assert_eq!(sink.events, ["params 760 12", "stream 8", "embedding 4"]);
}

#[test]
fn a_refused_token_start_writes_nothing() {
    for (token, stream, embedding, pos, want) in [
        // The padding rows inside the table, first, middle, last; the first row above the table.
        (248_070u32, 8usize, 4usize, 0usize, "sampler vocabulary"),
        (248_100, 8, 4, 0, "sampler vocabulary"),
        (248_319, 8, 4, 0, "sampler vocabulary"),
        (248_320, 8, 4, 0, "sampler vocabulary"),
        (u32::MAX, 8, 4, 0, "sampler vocabulary"),
        (1, 7, 4, 0, "start stream has 7"),
        (1, 8, 5, 0, "PLE embedding has 5"),
        (1, 8, 4, usize::MAX, "position does not fit an i32"),
    ] {
        let mut sink = Recorder::default();
        let result = write_start(
            &mut sink,
            &SHAPE,
            token,
            pos,
            &vec![0.0; stream],
            &vec![0.0; embedding],
        );
        match result {
            Err(EngError::BadRequest(message)) => {
                assert!(message.contains(want), "{message} (wanted {want})");
                if want == "sampler vocabulary" {
                    assert!(message.contains(&format!("token id {token}")), "{message}");
                }
            }
            other => panic!("{other:?}"),
        }
        assert!(sink.events.is_empty(), "{:?}", sink.events);
    }
    // The last id the sampler can draw is accepted.
    let mut sink = Recorder::default();
    write_start(&mut sink, &SHAPE, 248_069, 0, &[0.0; 8], &[0.0; 4]).unwrap();
    assert_eq!(sink.events.len(), 3);
}

#[test]
fn global_rows_adds_the_start_of_each_table_and_refuses_a_missing_table() {
    let shard_rows = [10u64, 5, 7];
    assert_eq!(
        global_rows(&[(0, 3), (1, 0), (2, 6), (1, 4)], &shard_rows).unwrap(),
        [3, 10, 21, 14]
    );
    assert!(global_rows(&[(3, 0)], &shard_rows).is_err());
    assert!(global_rows(&[(0, 0)], &[u64::MAX, 1]).is_err());
}

/// A reader whose runs sleep, so reads submitted together are in flight together, and whose
/// values name the row (`row as f32` repeated), so a block can be checked without a table.
fn named_reader(threads: usize, delay: Duration) -> RowReader {
    RowReader::with_reader(
        Arc::new(move |rows: &[u64]| {
            std::thread::sleep(delay);
            Ok(rows
                .iter()
                .flat_map(|&row| std::iter::repeat_n(row as f32, ROW_DIM))
                .collect())
        }),
        threads,
    )
    .unwrap()
}

#[test]
fn a_split_gives_every_thread_a_run_and_the_runs_cover_the_rows_once() {
    let reader = named_reader(4, Duration::ZERO);
    for n in [1u64, 3, 4, 5, 7, 33, 100] {
        let rows: Vec<u64> = (0..n).map(|i| i * 3 + 1).collect();
        let block = reader.read(&rows).unwrap();
        assert_eq!(block.ids(), rows, "n = {n}");
        for &row in &rows {
            assert!(block.get(row).unwrap().iter().all(|v| *v == row as f32));
        }
    }
}

#[test]
fn reads_in_flight_together_spread_over_the_pool_instead_of_queuing_on_the_first_workers() {
    // 8 workers, 4 rows per read: the parts of read A sit on workers 0-3 and the parts of read B on
    // workers 4-7, so both finish after one delay. Without the rotation B queues behind A and
    // needs two delays.
    let delay = Duration::from_millis(300);
    let reader = named_reader(8, delay);
    let start = std::time::Instant::now();
    let a = reader.submit(&[1, 2, 3, 4]).unwrap();
    let b = reader.submit(&[5, 6, 7, 8]).unwrap();
    let (b, a) = (b.wait().unwrap(), a.wait().unwrap());
    let elapsed = start.elapsed();
    assert_eq!((a.len(), b.len()), (4, 4));
    assert!(
        elapsed < delay * 3 / 2,
        "{elapsed:?} for two reads of one delay each"
    );
}

#[test]
fn a_pending_read_completes_after_the_reader_is_dropped() {
    let reader = named_reader(3, Duration::from_millis(20));
    let pending = reader.submit(&[10, 11, 12, 13, 14]).unwrap();
    drop(reader);
    let block = pending.wait().unwrap();
    assert_eq!(block.ids(), [10, 11, 12, 13, 14]);
}

#[test]
fn a_panicking_reader_thread_fails_the_read_and_every_later_submit_on_that_worker() {
    // One worker, so every submit lands on it. The first run panics; the worker ends.
    let tripped = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&tripped);
    let reader = RowReader::with_reader(
        Arc::new(move |rows: &[u64]| {
            if rows.contains(&13) && !flag.swap(true, Ordering::SeqCst) {
                panic!("test: the reader panics on row 13");
            }
            Ok(vec![0.0; rows.len() * ROW_DIM])
        }),
        1,
    )
    .unwrap();
    match reader.read(&[12, 13]) {
        Err(EngError::Checkpoint(message)) => assert!(message.contains("ended"), "{message}"),
        other => panic!("{other:?}"),
    }
    // The worker leaves its loop when the panic unwinds; its queue closes a moment after the
    // reply channel does, so poll for the failure instead of assuming the order.
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    loop {
        match reader.submit(&[1]) {
            Err(EngError::Checkpoint(message)) => {
                assert!(message.contains("has ended"), "{message}");
                break;
            }
            Ok(pending) => {
                // Queued before the worker's receiver closed: the read cannot be served.
                assert!(matches!(pending.wait(), Err(EngError::Checkpoint(_))));
            }
            Err(other) => panic!("{other:?}"),
        }
        assert!(
            std::time::Instant::now() < deadline,
            "the dead worker was never reported"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
}

fn percentile(values: &mut [f64], p: f64) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    values[((values.len() as f64 - 1.0) * p).round() as usize]
}

/// The full pack's table: the same prompt hashes to the same rows and decodes to the same bits as
/// the fixture, and the production reader's per-token latency and the time of 256- and 4096-token
/// chunks (plan, read plus decode, embedding assembly) are printed. Run with `--ignored
/// --nocapture`; the pack must be on local storage. The test drops no pages: a "cold" figure
/// relies on fresh pseudo-random rows over 30 GiB rarely being resident, so check `fincore` of the
/// file before the run when the figure matters. "Warm" is the same rows read again at once.
#[test]
#[ignore = "reads the 30.4 GiB n-gram table of the full pack: FIFI_FLASH_NEXT_EXL3_PACK"]
fn the_full_pack_table_serves_the_recorded_rows_and_reports_read_latency() {
    use std::time::Instant;

    let dir = std::env::var("FIFI_FLASH_NEXT_EXL3_PACK")
        .expect("FIFI_FLASH_NEXT_EXL3_PACK names the full pack directory");
    // The guarded opener of the loader (`FlashPackFiles` opens the same file the same way).
    let file = Arc::new(
        NgramFile::open(
            &Path::new(&dir).join("ngram_embedding.safetensors"),
            &LocalGuard::new().unwrap(),
            KEY,
            LAYOUT,
        )
        .unwrap(),
    );
    let hash = Hash::of(&file);
    let (unit, key) = ([()], ());
    let weights = hash.weights(&unit, &key);
    let shape = PleShape::new(dims()).unwrap();
    let reader = reader(&file, DEFAULT_THREADS);

    // Same revision as the fixture: the same rows, the same bits.
    let chunk = plan_chunk(&shape, &weights, &[], &PROMPT).unwrap();
    assert_eq!(chunk.all(), recorded_rows());
    let block = reader.read(&chunk.all()).unwrap();
    for (index, &row) in recorded_rows().iter().enumerate() {
        assert_eq!(
            bits(block.get(row).unwrap()),
            recorded_values(index),
            "row {row}"
        );
    }

    // Decode latency: 400 tokens of pseudo-random ids, history carried, cold then warm.
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        state.wrapping_mul(0x2545_F491_4F6C_DD1D)
    };
    let tokens: Vec<u32> = (0..400).map(|_| (next() % 248_000) as u32).collect();
    let chunk = plan_chunk(&shape, &weights, &[], &tokens).unwrap();
    let (mut cold, mut warm) = (Vec::new(), Vec::new());
    for rows in &chunk.per_token {
        for out in [&mut cold, &mut warm] {
            let start = Instant::now();
            let block = reader.submit(rows).unwrap().wait().unwrap();
            let embedding = block.embedding(rows, 1.0).unwrap();
            out.push(start.elapsed().as_secs_f64() * 1e6);
            assert_eq!(embedding.len(), shape.embed());
        }
    }
    println!(
        "token (16 rows) reader threads={DEFAULT_THREADS}, n={}: cold p50 {:.0} us p99 {:.0} us; warm p50 {:.0} us p99 {:.0} us",
        chunk.per_token.len(),
        percentile(&mut cold, 0.5),
        percentile(&mut cold, 0.99),
        percentile(&mut warm, 0.5),
        percentile(&mut warm, 0.99)
    );

    // Prefill chunks: all ids planned up front, one sorted read, then the embedding of every
    // token assembled from the block, which is what a chunk feed does.
    for count in [256usize, 4096] {
        let tokens: Vec<u32> = (0..count).map(|_| (next() % 248_000) as u32).collect();
        let t0 = Instant::now();
        let chunk = plan_chunk(&shape, &weights, &[], &tokens).unwrap();
        let planned = t0.elapsed();
        let t1 = Instant::now();
        let block = reader.read(&chunk.all()).unwrap();
        let read = t1.elapsed();
        let t2 = Instant::now();
        let mut checksum = 0.0f32;
        for rows in &chunk.per_token {
            checksum += block.embedding(rows, 1.0).unwrap()[0];
        }
        let assembled = t2.elapsed();
        println!(
            "chunk of {count} tokens: {} distinct rows; plan {:.1} ms, read+decode {:.1} ms, \
             embeddings {:.1} ms, total {:.1} ms (checksum {checksum})",
            block.len(),
            planned.as_secs_f64() * 1e3,
            read.as_secs_f64() * 1e3,
            assembled.as_secs_f64() * 1e3,
            t0.elapsed().as_secs_f64() * 1e3
        );
    }
}

#[test]
fn the_budget_line_is_no_reserved_page_cache_and_a_staging_bound_per_chunk() {
    assert_eq!(NGRAM_PAGE_CACHE_RESERVE_BYTES, 0);
    // 4096 tokens x 16 rows: 42 MB of decoded values, 44 MB with ids and the run copies.
    assert_eq!(ngram_staging_bytes(65_536), 65_536 * 656);
    assert!(ngram_staging_bytes(65_536) < 45 << 20);
    assert_eq!(ngram_staging_bytes(0), 0);
}
