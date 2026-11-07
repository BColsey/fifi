# Flash-Next n-gram table: read latency on the real file

Measured 2026-11-07 on the real table, `ngram_embedding.safetensors` of the 3.05 bpw pack (30.4 GiB,
320,001,536 rows of 102 bytes, rev 69e33439), on `/home` (btrfs, Samsung 990 PRO, PCIe 4.0 x4,
mount options `noatime,ssd,discard=async,autodefrag`). `fincore` showed 0 bytes of the file in the
page cache after the bench runs (read at the terminal, not logged) and before and after each
production-reader run (logged in `results.txt`).

## Method

- Program: `doc/flash-next-ngram-bench/ngram_bench.rs` (raw output in `results.txt`; `ple_matvec_bench.rs` is the PLE host matvec test) (std only, no profiling, built with `rustc -O -C target-cpu=native`),
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

## Per-row reads (queue depth = reader threads), two runs. Read only, no decode

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

## One decode token (16 random rows), two runs, 2,000 tokens each. Read only, no decode

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

## One prefill chunk (sorted, de-duplicated, pool of N threads). Read only, no decode

| tokens | rows (distinct) | threads | cold | warm |
|---|---|---|---|---|
| 256 | 4,096 | 16 | 14.9 ms (275k rows/s) | 1.0 ms |
| 256 | 4,096 | 32 | 10.4 ms (395k rows/s) | 1.3 ms |
| 4096 | 65,532 | 16 | 226 ms (290k rows/s) | 12.8 ms |
| 4096 | 65,528 | 32 | 142 ms (461k rows/s) | 12.8 ms |

These are raw reads: the bench does no `decode_row` and no embedding assembly. Through `RowReader`
(read + decode + embeddings for every token) a 4096-token chunk costs 242-244 ms (table below), so
a 32K prompt is eight chunks and about 2 s of cold n-gram work, read one chunk ahead of its compute
(about 3 s per chunk by the research notes). Serial per-row reads at 162 us (the old Kingston
figure) would take 85 s; at the measured 52 us they would still take 27 s.

## What it means for the design

- The old 162 us single-read figure was the Kingston drive. The real file on the 990 PRO is 52 us
  cold at QD1 under load.
- Decode: submit the 16 reads as soon as the token is known (the rows depend on the token and the
  last two tokens only) and wait for them at the PLE layer; the 160-320 us then overlap layer 0.
- Prefill: plan the whole chunk first, one sorted read at 32 threads.
- Page cache: nothing needs reserving (`NGRAM_PAGE_CACHE_RESERVE_BYTES = 0` in `feeds.rs`). The cold
  cost is small enough that a hit rate matters little; the kernel keeps what fits. Transient
  staging is `ngram_staging_bytes(rows)`: 42 MB for a 4096-token chunk. The trace study that would
  size a reserve (access distribution on real transcripts) has not been run.
- `mlock` of the table is not available (`RLIMIT_MEMLOCK` 8 MiB); no user-space row cache is built.

## The production reader on the same file

`feeds::tests::the_full_pack_table_serves_the_recorded_rows_and_reports_read_latency` (ignored; run
with `FIFI_FLASH_NEXT_EXL3_PACK=<pack> ... --ignored --nocapture`) drives `RowReader` with 32 threads
through `NgramFile::read_row` and `NgramDecoder::decode_row`: 400 random tokens (n = 400, so p99 is
the fourth largest sample) with the 2,560-value embedding assembly, then chunks of 256 and 4096
tokens. Release build, CPU only, `taskset -c 16-23`, load average 7, three runs. The test drops no
pages itself; I dropped the file's pages with `posix_fadvise(DONTNEED)` before each run and `fincore`
showed 0 resident before every run (`results.txt`). Within a run, "cold" is fresh random rows and
"warm" is the same 16 rows read again at once.

| case | run 1 | run 2 | run 3 |
|---|---|---|---|
| token, 16 rows, cold p50 / p99 | 200 / 323 us | 163 / 255 us | 160 / 238 us |
| token, 16 rows, warm p50 / p99 | 36 / 49 us | 41 / 63 us | 36 / 54 us |
| chunk of 256 tokens (4,096 rows): read+decode, total | 14.9, 15.3 ms | 14.9, 15.3 ms | 15.2, 15.6 ms |
| chunk of 4096 tokens (65,529 rows): plan | 1.9 ms | 1.9 ms | 1.7 ms |
| chunk of 4096 tokens: read+decode | 233.6 ms | 230.3 ms | 232.5 ms |
| chunk of 4096 tokens: 4096 embeddings | 8.8 ms | 9.5 ms | 9.3 ms |
| chunk of 4096 tokens: total | 244.2 ms | 241.7 ms | 243.6 ms |

An earlier set of two runs under load 22, before parts were rotated over the workers, gave cold p50
145 us / p99 268-385 us and warm p50 54 us. The warm token is hand-off plus decode (about 2 us per
row), not I/O. A debug build of the same test reports 266 us cold and 197 us warm at p50: the decode
loop needs an optimized build. The same rows decode to the recorded bits of the committed fixture.

## The PLE key/value matvec on the host (why the projections run on the card)

`doc/flash-next-ngram-bench/ple_matvec_bench.rs`: both projections (12,800 x 2,560, 33 M multiply-adds,
65 MiB as F16 or 131 MiB as F32) on `taskset -c 16-23`, 40 calls each, load 13-20: F16 on 8 threads
min 1.57 ms, p50 2.62 ms, p99 7.6 ms; F32 on 8 threads min 2.72 ms, p50 3.41 ms; F16 on 1-4 threads
p50 4.1-5.4 ms, p99 6-14 ms on 2-16 threads (61-70 ms on one). The 2.6-5 ms figures were taken under that load; the best case
is 1.57 ms with all 8 pinned cores busy and 65 MiB read from memory per token. The device route is
chosen for that traffic and those cores, and because the host product must finish while the card
runs layers 0 and 1 (about 1 ms); the kernel time is the figure measured with the kernel itself, and the route's own time
(pageable copies, two launches, a 44 KiB read-back) is not measured yet.
