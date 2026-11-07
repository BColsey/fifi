# Flash-Next n-gram table: read latency on the real file

Measured 2026-11-07 on the real table, `ngram_embedding.safetensors` of the 3.05 bpw pack (30.4 GiB,
320,001,536 rows of 102 bytes, rev 69e33439), on `/home` (btrfs, Samsung 990 PRO, PCIe 4.0 x4,
mount options `noatime,ssd,discard=async,autodefrag`). The page cache held 0 bytes of the file before
and after every run (`fincore`).

## Method

- Program: `doc/flash-next-ngram-bench/ngram_bench.rs` (raw output in `results.log`; `ple_matvec_bench.rs` is the PLE host matvec test) (std only, no profiling, built with `rustc -O -C target-cpu=native`),
  run as `taskset -c 16-23 ./ngram_bench <file> qd|token|chunk`. Buffered `pread` of 102 bytes
  (`FileExt::read_exact_at`, the call `NgramFile::read_row` makes) at uniformly random rows with
  `posix_fadvise(POSIX_FADV_RANDOM)` on the descriptor. Random uniform rows are the worst case: real
  text repeats rows.
- Cold: rows not read before in the run (a fresh random set per case), read once. Warm: the same
  rows read again immediately after (blocks of 256 rows per thread, or token by token), so the pages
  are resident. After each case the touched pages were dropped with `posix_fadvise(DONTNEED)` on
  their own 4 KiB ranges. No system-wide cache drop.
- Pool cases use persistent threads fed by channels, as `RowReader` does. A 16-row token is split
  over the threads; the time is dispatch to the last reply.
- The machine was busy with other lanes during the runs: load average 12-20, I/O pressure
  `some avg10` 60% (`/proc/pressure/io`), memory pressure 10%, 29 GiB of swap in use. The cold
  figures therefore include contention for the drive and the CPU. Treat them as a conservative
  bound. A first run of the "warm" pass after a full cold pass at 40k rows missed the cache (pages
  evicted under the pressure); the numbers below come from the interleaved runs.

## Per-row reads (queue depth = reader threads), two runs

| QD | cold p50 us | cold p99 us | cold max us | warm p50 us | warm p99 us |
|---|---|---|---|---|---|
| 1 | 52.4 / 51.5 | 72.5 / 66.2 | 296 / 2415 | 0.7 / 0.7 | 51 / 45 |
| 4 | 49.6 / 50.3 | 71.9 / 72.3 | 1458 / 1489 | 0.7 / 0.9 | 1.3 / 62 |
| 16 | 55.8 / 55.0 | 101.8 / 103.8 | 1285 / 2112 | 0.8 / 0.8 | 1.8 / 1.8 |
| 32 | 65.2 / 63.9 | 262.9 / 336.5 | 1782 / 2075 | 1.0 / 1.0 | 12.9 / 5.7 |

Aggregate cold throughput (cold plus warm reads over wall time, so about twice the cold rate): QD1
35k reads/s, QD4 115-154k, QD16 520k, QD32 700-800k. Default read-around (no `FADV_RANDOM`) at QD1:
cold p50 57.4 us, p99 72.6 us, against 52 us with `FADV_RANDOM`. The 10% gap does not justify a
new method on `NgramFile`, so the reader does not set the advice.

## One decode token (16 random rows), two runs, 2,000 tokens each

| reader threads | cold p50 us | cold p99 us | warm p50 us | warm p99 us |
|---|---|---|---|---|
| 1 (inline) | 837 / 817 | 1350 / 2356 | 7.7 / 7.5 | 14 / 15 |
| 4 | 247 / 220 | 492 / 364 | 17.9 / 12.3 | 37 / 21 |
| 8 | 147 / 139 | 302 / 232 | 18.5 / 18.6 | 53 / 32 |
| 16 | 111 / 109 | 336 / 324 | 34.6 / 31.1 | 78 / 52 |

Cold per-token cost on one thread is 16 times the per-row figure. With 8 to 16 pool threads it is
110-150 us p50 and 230-340 us p99 (about 1% of a 20-35 ms decode token). Warm, the pool costs more
than the inline loop (hand-off 10-30 us against 7.7 us for 16 resident rows): a decode token whose
rows are all resident is served faster inline, which `RowReader` does not do yet.

## One prefill chunk (sorted, de-duplicated, pool of N threads)

| tokens | rows (distinct) | threads | cold | warm |
|---|---|---|---|---|
| 256 | 4,096 | 16 | 14.9 ms (275k rows/s) | 1.0 ms |
| 256 | 4,096 | 32 | 10.4 ms (395k rows/s) | 1.3 ms |
| 4096 | 65,532 | 16 | 226 ms (290k rows/s) | 12.8 ms |
| 4096 | 65,528 | 32 | 142 ms (461k rows/s) | 12.8 ms |

A 32K prompt is eight 4096-token chunks: 1.1-1.8 s of cold reads in total, read one chunk ahead of
its compute (about 3 s per chunk by the research notes). Serial per-row reads at 162 us (the old
Kingston figure) would take 85 s; at the measured 52 us they would still take 27 s.

## What it means for the design

- The old 162 us single-read figure was the Kingston drive. The real file on the 990 PRO is 52 us
  cold at QD1 under load.
- Decode: submit the 16 reads as soon as the token is known (the rows depend on the token and the
  last two tokens only) and wait for them at the PLE layer; the 110-340 us then overlap layer 0.
- Prefill: plan the whole chunk first, one sorted read at 32 threads.
- Page cache: nothing needs reserving (`NGRAM_PAGE_CACHE_RESERVE_BYTES = 0` in `feeds.rs`). The cold
  cost is small enough that a hit rate matters little; the kernel keeps what fits. Transient
  staging is `ngram_staging_bytes(rows)`: 42 MB for a 4096-token chunk. The trace study that would
  size a reserve (access distribution on real transcripts) has not been run.
- `mlock` of the table is not available (`RLIMIT_MEMLOCK` 8 MiB); no user-space row cache is built.

## The production reader on the same file

`feeds::tests::the_full_pack_table_serves_the_recorded_rows_and_reports_read_latency` (ignored; run
with `FIFI_FLASH_NEXT_EXL3_PACK=<pack> ... --ignored --nocapture`) drives `RowReader` with 32 threads
through `NgramFile::read_row` and `NgramDecoder::decode_row`, 400 random tokens, including the
2,560-value embedding assembly. Release build, `taskset -c 16-23`, load average 22, the file's pages
dropped before each run (two runs):

| case | p50 | p99 |
|---|---|---|
| token, 16 rows, cold | 145 / 145 us | 385 / 268 us |
| token, 16 rows, warm (re-read at once) | 54 / 55 us | 460 / 204 us |
| chunk of 256 tokens, 4,096 distinct rows, cold | 17.7 / 17.4 ms | - |

The warm token is hand-off plus decode (about 3 us per row at 16 rows), not I/O. A debug build of the
same test reports 266 us cold and 197 us warm at p50: the decode loop needs an optimized build. The
same rows decode to the recorded bits of the committed fixture.
