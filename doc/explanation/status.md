# Current status

**Goal: 128K to 256K tokens of context for every model, with EXL3 as the primary weight format.** The context rows show the measured admission limits and boot results.

This page records measurements by model and backend.
It holds measured facts only. Every row gives the date, the conditions and the receipt it comes from, and anything that is an estimate says so.
Numbers move with every merged change, so the date on each row matters more than the order of the rows.

A 31 Oct 2026 snapshot was recorded at `87bebce43`. Later entries have their own dates and sources.

How to read the rows:

- **Bare** means the server was started fresh for the run, prefix reuse was off, and the prompt was sent once.
- **Boots** is the number of fresh server starts behind a figure. A figure from one boot is a single run and carries about 4% day-to-day noise; a range lists the individual runs.
- **qmd** is a background search indexer that sometimes runs on the test machine at low priority. A row says whether its run recorded it. "Unverified" means the run's logs hold no record either way.
- **First token** is the server's own interval: from admission into the engine queue to the first batch of generated tokens (`time_to_first_token_ms` in the request telemetry). Time spent waiting in that queue is included; logging starts after the request is submitted to it, so earlier time and network time are excluded. A row that was measured by a client from sending the request is labelled "client-observed", and a row whose result file does not say which interval it used is labelled "interval not recorded".
- **Units**: GB is 10^9 bytes; MiB and GiB are binary (2^20 and 2^30 bytes). Rows that quote nvidia-smi use MiB.
- **Profile** is the KV cache setting of the run. Where a run did not record it, the row says "profile not recorded".
- Receipts are kept in the project's measurement archive, not in this repository. Each source names the receipt file, or its kind where the archive file name is an internal tag, and, in brackets, the first eight characters of its SHA-256.
- Compare rows only when their conditions match; for comparisons against other engines, see [Benchmarks](benchmarks.md).

## Qwen3.6-35B-A3B on an RTX 3090 (CUDA)

The default profile stores the KV cache in 8 bits; a 16-bit profile is available. See [Run Qwen3.6](../how-to/qwen36.md). The rows in the first table come from one run with the `int8g32` KV profile and a 180,224-token context limit, as its load line records. The binary is not recorded.

| Measure | Value | Date | Conditions | Source |
| --- | --- | --- | --- | --- |
| First token, cold 4K prompt | 1.01 and 1.02 s | 29 Oct 2026 | 4,534-token prompt, 128 generated tokens, one boot per row; qmd unverified | `legA.receipt.txt` (2ce20972), `legB.receipt.txt` (1163f54f) |
| Decode speed | 164.5 to 173.1 tokens/s | 29 Oct 2026 | 6 requests of 128 tokens each, prompts of 4,534, 4,590 and 1,181 tokens, both receipts | same receipts |
| First token with prefix reuse on | 0.14 s (4,096 of 4,590 prompt tokens reused) | 29 Oct 2026 | second request of the run, sharing a 4,527-token prefix with the first; the receipt's own gate line reports a failure for this leg, and the figure is read from the per-request telemetry rows | `legB.receipt.txt` (1163f54f) |

Context length limits:

| Context limit | Result | Date | Conditions | Source |
| --- | --- | --- | --- | --- |
| 180,224 tokens | loads the weights and passes admission | 29 Oct 2026 | `int8g32` profile | `A180224.log` (753a9550) |
| 262,144 tokens | did not boot: out of memory while allocating | 29 Oct 2026 | 1 attempt, without the pinned prefix; profile not recorded (the load failed before a load line) | `B262144.log` (a91daf98) |
| 262,144 tokens | did not boot: the 512-row prefill staging does not fit the budget | 29 Oct 2026 | 1 attempt, with the pinned prefix on; profile not recorded | `C262144prefix.log` (cdf6ffb1) |

### Qwen3.6-35B-A3B, EXL3 4.00 bpw (merged, behind `weight_format`)

Qwen3.6-35B-A3B in the EXL3 4.00 bpw format on the same card (30 Oct 2026). The EXL3 serving stack is merged (tip `c00c145fa`) and is selected with `"weight_format": "exl3"`; the GGUF file stays the default. The rows below are as measured on the tree committed as `c8b37bb8f`, an earlier state of that stack. The engine loads the whole pack and serves coherent text (a 29-token prompt gets a coherent three-sentence answer). The `int8g32` KV profile; first token and decode are the server's telemetry; one boot per server log (the two EXL3 prompts of `srv-c8k.log` share a boot, as do the two GGUF prompts). The GGUF engine was measured the same day on the same machine. Generated tokens: EXL3 83 and 60 (`srv-c8k.log`) and 32 (`srv-d262k.log`); GGUF 88 and 64.

| Run | Context | Weights resident | First token | Decode | Receipt |
| --- | ---: | ---: | ---: | ---: | --- |
| EXL3 4.00 bpw, 29-token prompt | 8,192 | 18.82 GB | 0.445 s | 68.63 tokens/s | `srv-c8k.log` (55622fda) |
| EXL3 4.00 bpw, 5,222-token prompt | 8,192 | same | 77.32 s | 66.69 tokens/s | same log |
| EXL3 4.00 bpw, 29-token prompt | 262,144 admitted | 18.82 GB | 0.445 s | 68.6 tokens/s | `srv-d262k.log` (96229ebb) |
| GGUF engine, 29-token prompt | 8,192 | 20.69 GB | 0.054 s | 164.0 tokens/s | `srv-gguf.log` (035d80a1) |
| GGUF engine, 5,222-token prompt | 8,192 | same | 1.04 s | 163.04 tokens/s | same log |

The two EXL3 rows above were measured with prompt tokens processed one at a time (5,222 tokens took 77 s against 1.04 s on GGUF). Chunked prefill for the EXL3 pack was merged afterwards (the commits `b0600443c..1de94e65d`); the next table measures it. The EXL3 decode is about 42% of the GGUF engine's.

Chunked prefill on the merged tree (31 Oct 2026). Server built from the tip of that range, `1de94e65d` (binary sha8 14c8ab6c), Qwen3.6 EXL3 4.00 bpw, the default KV setting (`kv_dtype` set to `f32`, the CUDA default, which stores the cache as FP16), no chunk switch set, the 5,213-token prompt file `prompt-5k.txt` (f0c044b1), 32 generated tokens, one boot per context. First token and decode are the server's telemetry. The host ran at 19% CPU on average (79 samples, 28% at most) with no builds running, so these rows are not CPU-contended (`fp-vmstat.log`, ea57efae). The server log shows the 5,213-token prompt split into chunks of exactly 2,047, 2,048, 1,024 and 93 rows.

| Run | Context | First token | Decode | Receipt |
| --- | ---: | ---: | ---: | --- |
| 29-token prompt | 8,192 | 161.2 ms | 65.71 tokens/s | `srv-fp-base-8192.log` (07cda9fe) |
| 5,213-token prompt, two requests | 8,192 | 2,892.9 and 2,979.9 ms | 60.15 and 61.61 tokens/s | same log |
| 29-token prompt | 32,768 | 109.4 ms | 62.35 tokens/s | `srv-fp-base-32768.log` (de013c6b) |
| 5,213-token prompt, two requests | 32,768 | 3,014.8 and 2,971.0 ms | 63.22 and 63.54 tokens/s | same log |
| 5,213-token prompt, two requests, `Q36_CUDA_CHUNK_PREFILL=0` set | 8,192 | 2,923.1 and 2,930.2 ms | 64.72 and 64.5 tokens/s | `srv-fp-base-8192-nochunk.log` (b955bc59) |

At the measured tree the switch `Q36_CUDA_CHUNK_PREFILL=0` changed nothing with this KV setting: it only turned off the chunk path of the `q8` profile. The run summary is `q36pf-fp1.out` (219af0cd).

Rollback switch and correctness at the default KV (31 Oct 2026). Server with `Q36_CUDA_CHUNK_PREFILL=0` switching the chunk path off at this KV (binary sha8 f0076788), same prompt file, 8,192 context, one boot per row, 32 generated tokens, host CPU busy 20% on average (117 samples, 35% at most; `rb-vmstat.log`).

| Run | First token | Decode | Output text (sha8) | Receipt |
| --- | ---: | ---: | --- | --- |
| default KV, no switch | 2,716.4 ms | 68.06 tokens/s | 5da6371f | `srv-rb-fp16-default.log`, summary `q36pf-rb2.out` |
| default KV, `Q36_CUDA_CHUNK_PREFILL=0` | 76,521.5 ms | 67.33 tokens/s | 5da6371f | `srv-rb-fp16-off.log`, same summary |
| `q8` KV, no switch | 2,948.5 ms | 67.19 tokens/s | d57800c7 | `srv-rb-q8-default.log`, same summary |

With the switch set to 0 the 5,213-token prompt runs one token at a time and the 32 generated tokens equal those of the chunked run. Against the one-token path, on the real pack at the default KV (2,048-row first chunk), the next-token distribution after a 5,222-token prompt differs by 5.2e-4 nats (KL) and the worst of 33 teacher-forced steps by 4.0e-3 nats, with two greedy flips, both where the two best logits were within 0.1 of each other (the first step: 0.001); a 3,300-token prompt gives 3.3e-5 and 1.9e-3 with no flip (`q36pf-fp5.out`, 298786b8, test binary sha8 eccd57db). Quality against the GGUF engine is not measured yet; the registered quality check applies.

## Qwen3.6-35B-A3B on Apple Silicon (M4 Max)

Profile not recorded in the receipts below. All rows: 28 Oct 2026, engine commit `66157b358`, a 30K-token coding replay of 13 turns, temperature 0, 2 runs per engine in alternating order. Source: the comparison report (f5e0d1b6) and `w34-summary.json` (bed32d97).

| Measure | Value | Conditions |
| --- | --- | --- |
| Summed first-token time, turns 1 to 12 (client-observed) | 28.4 s (llama.cpp 31.3 s) | sum of the per-turn medians over the 2 runs, measured by the replay client |
| Decode speed per turn, turns 1 to 12 | 60.4 to 72.0 tokens/s | first fifi run |
| Peak memory (resident set) | 12,948.0 and 15,206.9 MiB (llama.cpp 22,527.1 and 22,831.8 MiB) | one value per run |
| Turn 0 first token (client-observed) | 5.70 s (llama.cpp 0.075 s) | median of 2 runs, an 18-token prompt on a fresh start, so it includes one-time warmup |

## Flint on an RTX 3090 (CUDA)

Weights are the EXL3 4.0 pack. The KV cache format is a profile setting, `f16` or `q8`; both are implemented and measured below. Prefix reuse is implemented for the `f16` profile only. Details are in [Run Flint](../how-to/flint.md).

Cold 4K prompt, `f16` profile, bare (these token-major figures were measured before the job-table prefill became the default, so they describe `FIFI_EXL3_LAYER_MAJOR=off`; the default row is in the switch table below):

| Measure | Value | Date | Conditions | Source |
| --- | --- | --- | --- | --- |
| First token | 43.3, 44.2 and 44.2 s | 30 Oct 2026 | 3 boots, 4K-token prompt, binary `287cdd69`; qmd unverified | `prefill-bare-run2.log` (0e4b61a1) |
| Decode speed | 35.0, 35.9 and 35.5 tokens/s | 30 Oct 2026 | same runs | same log |
| Prompt processing cost | 11.24 ms per prompt row | 29 Oct 2026 | production A/B of the packed kernels, mean of 2 runs, profile not recorded | `AB-RESULT.md` (44e36b5c) |
| Warm request sharing a 3,999-token prefix, prefix reuse on | 0.71 s to first token (the first request of the same run, a miss, took 44.4 s) | 29 Oct 2026 | `f16` profile, context 65,536, 4,056-token prompt, 1 boot | `legB.receipt.txt` (f3a15478) |

Switches that move these numbers on the current main:

| Switch | Default | Measured effect | Date | Conditions | Source |
| --- | --- | --- | --- | --- | --- |
| `FIFI_EXL3_SHARED_IN_ROUTED` (the Qwen3.6 EXL3 decode step; unset, `0` and `1` against each other) | on for a Qwen3.6 EXL3 load (unset is on; `0` rolls back; a load that cannot run it falls back to the reference block and logs why); off when measured | decode 122.97 to 130.55 tokens/s (+6.2%) in the three-arm window and 120.20 to 128.93 (+7.3%) in the earlier two-arm window; `0` against unset 123.28 to 122.97 (no difference); 7 fewer kernel launches per MoE layer (17 to 10 graph nodes); output identical on all 80 replies of the three runs | 7 Nov 2026 | 8K prompt, 256 tokens greedy, q8 KV, capture on, 4.00 bpw pack, 325 W, boots interleaved (two-arm window: off, on, on, off, repeated; three-arm window: unset, 0, 1, 1, 0, unset, repeated), 4 boots per arm; speed differs by 2 to 4% between windows, so only ratios inside one window count | `ARMS.md` (three-arm window) and `AB.md` (two-arm window), binary `baa069da` |
| `FIFI_FLINT_TOKEN_HEALTH` | on (default since the 31 Oct 2026 change that made it so; `0` rolls back); off when measured | decode 35.49 to 37.92 tokens/s (+6.9%); first token 44.8 to 44.9 s (unchanged); output identical on all six boots | 30 Oct 2026 | 3 boots per arm, interleaved, 4K prompt, binary `3ac5af37`, profile not recorded; qmd running at low priority in both arms (recorded before and after each run) | `th-ab-off.log` (189da43a), `th-ab-on.log` (042f3fe2) |
| `FIFI_FLINT_TOKEN_HEALTH` (on against off, run that made it the default; KV device guard on in both arms) | on | 4K prompt decode means 47.35 to 50.30 tokens/s (+6.2%; runs 47.44 and 47.25 against 50.34 and 50.25); 22-token prompt 48.41 to 51.18 (+5.7%); first token at 4K 31.73 and 31.97 s against 31.52 and 31.92 s (unchanged within noise); output identical in all 8 requests (4 boots, 2 prompts each), and the 4K identity arm's text hash stays c935d745 with both switches on | 31 Oct 2026 | 2 boots per arm, alternating, bank of 224 experts per layer, 4,096 context, `f16` profile, same engine binary in both arms (wide single-row kernel on), host load average 4 to 15 | `tA1-083452` (98c3fc61), `tB1-090728` (bcb517b1), `tA2-091347` (2b3a34a6), `tB2-091441` (4c55c2eb); sha8 is of each boot's client log |
| `FIFI_FLINT_CUDA_FUSED_PUBLISH` | on (default since `8cb20f64f`; `0` rolls back) | decode 36.73 to 39.74 tokens/s (+8.2%); output identical on all six boots; first token not compared | 30 Oct 2026 | 3 boots per arm, interleaved, 4K prompt (4,000 tokens), `f16` profile, `FIFI_FLINT_TOKEN_HEALTH=1` in both arms, commit `e4f1672d5`, qmd running at low priority before and after (host load average about 17 during the run) | `fp-ab.log` (a77aa742) |
| `FIFI_FLINT_CUDA_KV_DEVICE_F16` | on (default since the 31 Oct 2026 change that made it so; `0` rolls back); off when measured | decode 41.70 to 43.47 tokens/s (+4.2%, means of 3 boots); first token 41.44 to 40.39 s (-2.5%); output identical on all six boots | 30 Oct 2026 | 3 boots per arm, interleaved, 4K prompt, `f16` profile, 65,536 context, `FIFI_FLINT_TOKEN_HEALTH=1` and the fused publish on in both arms, commit `f962f11f8`, qmd running, host load average rose from 7.8 to 13.2 during the run | `kvdevice-ab-run2.log` (1461f9ee) |
| `FIFI_FLINT_CUDA_KV_DEVICE_F16` (on against the host path, run that made it the default) | on | 4K prompt decode means 45.09 to 47.63 tokens/s (+5.6%; runs 45.16 and 45.02 against 47.53 and 47.73); 22-token prompt 45.63 to 47.19 (+3.4%); 13,295-token prompt 43.85 to 45.35 (+3.4%; runs 43.77 and 43.93 against 44.09 and 46.60); first token means 32.61 to 31.66 s at 4K (-2.9%) and 113.40 to 110.86 s at 13K (-2.2%); output identical in all 12 requests (8 boots: 4 at 4,096 context with the 22-token and 4K prompts, 4 at 32,768 context with the 13K prompt), and the 4K identity arm's text hash stays c935d745 | 31 Oct 2026 | 2 boots per arm, alternating, bank of 224 experts per layer, `f16` profile, wide single-row kernel (`c671b5d34`), host load average 10 to 15 | `kA1-082233` (663c67eb), `kB1-082649` (dbc1fb62), `kA2-082744` (e476ae69), `kB2-083031` (12645b63), `sA1-083529` (7835c37f), `sB1-084238` (15a52da4), `sA2-084452` (1f8384ba), `sB2-084710` (7b27ae1d); sha8 is of each boot's client log |
| `FIFI_EXL3_LAYER_MAJOR=1` (tile pass) | off (opt-in) | first token 41.49, 41.85 and 42.40 s with it off, 31.14, 30.87 and 30.71 s with it on (means 41.91 and 30.90 s, -26.3%); decode was 5.5% slower in this first run (38.14, 38.29 and 38.33 tokens/s off against 35.57, 35.32 and 37.55 on) but a repeat on the current main did not reproduce it (first token 41.28 s off, 30.19 s on, -26.8%; decode 38.72 off and 39.16 on for the cold request, 40.08 and 40.40 for a second 384-token request; 3 boots per arm; repeat result file 7585e43d), so no decode cost is established; **the quality check failed**: teacher-forced against the off path on 768 positions of the K16 text class, mean KL divergence 0.641 nats, worst position 18.4, and 142 of 768 argmax tokens (18.5%) differ, far above both pre-set pass bars (a mean of 0.005 with 7 flips, or 0.003 with 3 flips after the byte-exact repeat tightened it); the off path repeats byte for byte, so the difference is the switch, not noise; the error comes from running the routed experts as tall tiles over a whole prompt piece; a variant that runs them in 8-row blocks is bit-identical to the off path and decodes at the same speed (39.85 to 40.11 against 39.75 to 39.81 tokens/s), but it doubles the first-token time (41.29 s off, 82.80 s with it, 3 boots per arm), so it gives no gain; the tall-tile variant is parked pending a decision on the quality standard, the switch stays off, and the speed figures here are not a pending win | 30 Oct 2026 | 3 boots per arm in interleaved order (off-on-off, then on-off-on), 4,000-token prompt, greedy, 64 generated tokens (first run), `f16` profile, 65,536 context; both figures are the server's own telemetry; qmd not running; tested binary built from commit `4de73203b`; other switches not recorded in the summary | server-side result file of the layer-major A/B (9ca01729); quality: layer-major KL verdict file (cb4ec1af) and noise receipt (6611e3e7); 8-row-block variant: bounded-experts A/B verdict file (d9ea44f4) |
| `FIFI_EXL3_LAYER_MAJOR` (job-table pass `jobs`, the default on the `f16` profile, with the hi/lo split `FIFI_EXL3_TALL_HILO` on; `FIFI_EXL3_LAYER_MAJOR=off` rolls back to the token-major walk, `FIFI_EXL3_TALL_HILO=0` drops only the split) | `jobs` on `f16` | 4,000-token prompt first token 26.65 to 4.43 s (-83.4%), 16,002-token prompt 114.24 to 19.77 s (-82.7%); cold-request decode 51.4 to 47.4 tokens/s at 4K and 48.4 to 42.0 at 16K, 384-token follow-up level (51.4 to 51.1 and 49.9 to 50.7), because the pass holds 9 of the 225 expert slots (216 left; prefill misses 22,438 to 6,632 at 4K and 93,298 to 26,652 at 16K); peak VRAM 22,806 to 23,056 MiB; text differs from the token-major walk; quality: the default rests on the project owner's explicit override of the registered 32K gate on 7 Nov 2026, not on a gate pass; the registered check fails for jobs with the split (next-token loss +0.0239 nats, 95% upper bound +0.043 against +0.02; top-1 -0.447 points, lower bound -0.775 against -0.5) and for plain jobs (+0.0315; top-1 -0.400), and those verdicts stay FAIL; the gate cannot resolve effects of this size because its margin (0.02 nats) equals the run-to-run scatter of reordered runs (standard deviation about 0.02); the mean of four reordered runs of jobs with the split is +0.0038 nats against the referee and +0.0174 above the token-major walk's reordered runs (t 1.4, not separated); estimated cost: point estimate about +1.8 percent perplexity at 32K (exp(0.0174) - 1), 95 percent interval roughly -1.3 to +4.9 percent (four reordered runs against four, Welch t 1.4, so not separated from zero); the check covers a jobs prefill of only 7,423 (16K class) and 11,519 (32K class) rows, not the longer prompts the default now serves, and its copy and retrieval probes are empty at 32K; receipts: $SCRATCH/e22507c70/flint-prefill-gap/ctl/runs/v6-1/SUMMARY.md (2f24b0fd), report-h0.json (5f86851c) | 7 Nov 2026 | 3 boots per arm, `f16` profile, default 131,072 context, the same frozen engine binary `b67a6a85` in both arms | first-token table of the first A/B (`$SCRATCH/e22507c70/flint-prefill-gap/jobs-ab/runs/jobs-ab-1/AB.md`, 5815f6c4; `ab.json`, 8a1554f1) |
| `FIFI_EXL3_TALL_HILO` (hi/lo split of the tall projections of the jobs pass; on with `jobs`, `0` rolls back) | on with `jobs` | 4,000-token prompt first token 4.49 s (plain jobs) to 4.65 s (+3.6%); 16,002-token prompt 20.1 to 21.0 s (+4.5%); token-major walk 27.93 and 118.3 s, so the default is 6.0 and 5.6 times faster than it; cold-request decode (the first request after the prefill) is 9 to 11 percent below the token-major walk (4K 46.2 against 50.6 tokens/s, 16K 42.4 against 47.5) and the 384-token follow-up is level (50.7 against 51.3 and 49.0 against 49.8); the split itself changes decode by nothing against plain jobs; text differs from the plain jobs pass (it differs from the token-major walk in both) | 7 Nov 2026 | 2 to 3 boots per arm, interleaved arms (token-major, jobs, jobs plus split, jobs on the F32 SIMT path: 9.79 and 41.7 s), `f16` profile, default 131,072 context, one engine binary | four-arm table (`$SCRATCH/e22507c70/jobs-f16split/ab/runs/ab4-1/AB.md`, dabce651; `ab.json`, d8319393); quality basis: the row above |
| `FIFI_FLINT_CUDA_PACKED_POINTWISE` | on | decode 31.21 to 35.31 tokens/s (+13.1%); first token 59.12 to 44.97 s (-23.9%) | 29 Oct 2026 | 2 runs per arm, alternating order, 4K prompt, output identical, profile not recorded | `AB-RESULT.md` (44e36b5c) |

`q8` profile, tiled attention route (earlier build, packed kernels off in this comparison; first-token interval not recorded). Source for the three rows: the tiled-route A/B result file (680131a7), 29 Oct 2026, 1 pair of runs:

| Measure | Older kernel | Tiled route |
| --- | --- | --- |
| First token, 4K | 118.68 s | 64.85 s |
| First token, 32K (31,986 tokens) | 1,289.24 s | 633.75 s |
| Decode speed, 4K and 32K | 21.37 and 14.30 tokens/s | 30.38 and 24.22 tokens/s |

**Quality standard changed (30 Oct 2026): real-task non-inferiority with registered margins; KL is kept as a diagnostic.** Paths that change numerics (the `q8` KV profile and the layer-major prefill) are judged by three gates: correctness (hard); numerical characterization (KL divergence, argmax differences, layer drift: reported only); and application quality, a pass or fail by non-inferiority at the 95% bound, with a paired bootstrap clustered by document. The registered margins are: true-token negative log-likelihood no worse than +0.02 nats, top-1 accuracy no worse than -0.5 points, exact-copy probes no worse than -1.0 points, retrieval probes no worse than -2.0 points. Earlier KL results stay recorded as failures. The 9,216-position corpus and the 16K and 32K slices below were investigation data and do not count toward qualification; qualification uses fresh held-out text. Registration v7 (6 Nov 2026, future collections only) replaces the two exact-match probe gates with probe-span NLL gates (+0.019 nats each, for a 0.02-nat tolerated harm) because the exact-match gates were underpowered; earlier verdicts stay as registered under v6 and are not re-judged (`qualification/o1-gate-v7/`). Until the gates pass the `q8` profile stays opt-in and the layer-major prefill stays off; the 131,072-token default context of Flint CUDA uses the `f16` profile.

Quality of the `q8` profile against the `f16` profile, teacher-forced on a fixed text corpus (mean KL divergence per scored position, in nats; 768 scored positions; 30 Oct 2026; source the quality verdict file (24df17af)):

| Context | Tiled route | Older route | Notes |
| --- | --- | --- | --- |
| 16K | 0.654 | 0.739 | the tiled route passes the relative gate against the older route |
| 32K (same positions) | 0.983 | not run | 203 of 768 argmax tokens differ from the `f16` profile (`DUMP-REPORT.txt`, d3d5df5c) |

Quality of the `q8` profile against the `f16` profile on true text (the next token of the corpus, not agreement with the `f16` path), tiled route, paired by position with a bootstrap 95% interval (10,000 resamples), 30 Oct 2026. On 9,216 positions (classes S 6,144, M 1,536, L 768 and K16 768) the mean negative log-likelihood of the true token is 1.8168 nats against 1.8408 for `f16` (difference -0.0240, interval -0.0499 to +0.0018) and top-1 accuracy is 68.03% for both (difference 0.0000, interval -0.0040 to +0.0041), while 10.2% of the argmax tokens differ from the `f16` path. No difference in true-token quality is statistically significant at this size, but the check is not a proof of harmlessness. Per class, the difference in top-1 accuracy is +0.03 points (S), +0.72 (M), -0.13 (L) and -1.56 (K16, interval -3.52 to +0.39, the only negative estimate); argmax differences grow with depth, 6.5% at L, 9.3% at S, 10.4% at M, 21.0% at K16 and 26.4% at 32K (768 positions: NLL 3.2963 against 3.5045, accuracy 55.21% against 55.34%). A split by layer shows the late layers carry the sensitivity: `q8` on layers 0 to 24 with `f16` on 25 to 48 changes 7.7% of the argmax tokens (mean KL 0.036), the mirror 26.7% (mean KL 1.00), on the 32K positions (low-arm receipt 1558c1bd, high-arm receipt 1e288731). The comparison script's output is stored as the recomputed ground-truth table (9e76fa3b); the registered check remains the KL divergence against the `f16` path above.

Context that fits with every expert on the GPU (admission ceilings, found by search on the 24 GB card on 27 Oct 2026). Source: `CAPACITY-RECEIPT.json` (38bec5da). It records that the search tested admission only, not a served run. The `f16` figure is superseded: on 30 Oct 2026 the build at `e1698df1c` names 60,096 tokens as the largest `f16` context with the expert cache off (receipt 88a16a09, in the table below). The `q8` figure has not been measured again:

| Profile | Tokens |
| --- | ---: |
| `f16` (27 Oct build; 60,096 on 30 Oct, see above) | 45,315 |
| `q8` | 86,095 |

Default boot (30 Oct 2026, server built from the source merged as `e1698df1c`): with no `flint` object in the config, Flint serves a 131,072-token context on the `f16` profile and keeps only as many experts per layer on the GPU as the context allows (226 of 256, all 256 in pinned host memory); the startup line says so and `FIFI_FLINT_CUDA_EXPERT_CACHE=0` keeps every expert on the GPU instead. The `q8` row below is not a default: its config sets the profile and the context. One boot per row (N=1; the two 16K rows share one boot), prompts of 4,000 tokens unless stated, 128 generated tokens, first token and decode from the server's telemetry, process peak in nvidia-smi MiB. Each receipt cell names the run folder, the file and the first 8 hex of its SHA-256:

| Run | Context | Expert slots | First token | Decode | Process peak | Receipt |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| No `flint` object, 4K prompt (prefix reuse off) | 131,072 | 226, chosen from the context | 45.67 s | 38.15 tokens/s | 22,264 MiB | `leg1-default-4k` `RECEIPT.json` (68a59c83) |
| Context set to 32,768, this server | 32,768 | none (all 256 on the GPU) | 42.36 s | 38.85 tokens/s | 21,732 MiB | `leg2-explicit32k-new` `RECEIPT.json` (e9c064c6) |
| Context set to 32,768, previous server (commit `94069e91b`, main before the automatic bank and the 131,072 default) | 32,768 | none | 42.33 s | 39.29 tokens/s | 21,732 MiB | `leg3-explicit32k-base` `RECEIPT.json` (3f11f50e) |
| Expert cache off, 131,072 | 131,072 | none | refused at start: needs 25,271 MiB, 23,330 MiB free; the message names 60,096 tokens as the largest `f16` context that fits | not applicable | not applicable | `leg4-off-131072` `RECEIPT.json` (88a16a09) |
| Expert cache off, 32,768, 10-token prompt | 32,768 | none | 0.13 s | 39.11 tokens/s (8 tokens) | 21,732 MiB | `leg5-off-32768` `RECEIPT.json` (df98e201) |
| No `flint` object, real 16,002-token prompt, cold (prefix reuse on in the config) | 131,072 | 226 | 174.47 s | 39.04 tokens/s | 22,264 MiB | `leg6-default-real16k-warm` `RECEIPT.json` (3ad31ce7) |
| Same boot, second request with prefix reuse on (a different prompt of 16,502 tokens, 16,001 reused, 501 new) | 131,072 | 226 | 5.87 s | 39.04 tokens/s | 22,264 MiB (process peak of the whole boot) | `leg6-default-real16k-warm` `server.log` (78ea4d24) |
| `q8` profile and 262,144 context set in the config (prefill chunk 8, prefix reuse off); expert slots chosen from the context | 262,144 | 223 | 53.21 s | 34.96 tokens/s | 22,254 MiB | `leg7-q8-262144` `RECEIPT.json` (fe1b75ec) |

Prefix reuse was switched on in the config of the boot behind the two 16K rows. The cold 16,002-token first token (174.47 s) is about 30 times the second request's (5.87 s), but the two prompts differ: the second has 16,502 tokens and only 501 of them were processed. The quality of the `q8` profile at these depths is not measured.

Context window admitted with the expert cache on (`FIFI_FLINT_CUDA_EXPERT_CACHE=1`, set by hand in these runs, 30 Oct 2026; since `e1698df1c` the cache is chosen automatically). **These runs show the engine admits and serves the window; each sent a 4,000-token prompt and generated 128 tokens, so long prompts are not yet tested.** One boot per row, N=1, `f16` or `q8` profile as listed, greedy, no prefix reuse; first token and decode are the server's telemetry; the process peak is nvidia-smi MiB; the expert slots are how many of the 256 experts per layer stay in video memory:

| Context | Profile | Expert slots | First token | Decode | Peak video memory (process) | Expert misses per decoded token | Receipt |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 131,072 | `f16` | 224, fixed (merged code, commit `94069e91b`) | 42.02 s | 39.35 tokens/s | 22,120 MiB | not counted | `RECEIPT.json` of the base leg (38237419) |
| 131,072 | `f16` | 226, chosen from the context (commit `c1b2997f9`, merged as `f849d7eb9`; the miss counter is from commit `f2a78d294`, the same change built before merge) | 41.62 s | 39.42 tokens/s | 22,264 MiB | 8.12 | `RECEIPT.json` of the auto-slot leg (527bb3be) |
| 262,144 (the model limit) | `q8` | 223, chosen from the context (commit `c1b2997f9`, merged as `f849d7eb9`; the miss counter is from commit `f2a78d294`, the same change built before merge) | 53.81 s | 34.34 tokens/s | 22,254 MiB | 7.70 | `RECEIPT.json` of the 262K leg (36043d8d) |

The first two rows give identical text (sha256 prefix c935d745); the third differs because it uses the `q8` profile. Pinning the expert store takes about 18 to 21 GB of host memory (available memory fell from 50.9 to 32.3 GB on the first row) and moved only 0.1 to 0.3 GB to swap. With the expert cache off (`FIFI_FLINT_CUDA_EXPERT_CACHE=0`), the largest `f16` context is 60,096 tokens on the build at `e1698df1c` (receipt 88a16a09; the 27 Oct build found 45,315), and the `q8` ceiling found on 27 Oct was 86,095; the quality of the `q8` profile at these depths is not measured.

Real long prompts with the expert cache on (30 Oct 2026, token-major walk; the figures describe `FIFI_EXL3_LAYER_MAJOR=off`). The prompts are distinct repository documents and Rust source, sized with the model tokenizer; greedy, 128 generated tokens, `f16` profile, binary built from commit `8ba1d3d86` (sha256 prefix 52365abb), one boot per row (N=1). First token and decode are the server's telemetry; the host load average was 13 to 14 from other jobs during the runs; swap moved under 0.1 GB. An earlier 13,294-token run used a repeated sentence whose expert routing is unrepresentative and is not listed. The 16,002-token prompt also appears in the default-boot table above (174.47 s with 226 slots on the build at `e1698df1c`); the 173.4 s below is a different build (`8ba1d3d86`) with 224 slots.

| Prompt tokens | Context | Expert slots | First token | Decode | Expert misses per decoded token | Receipt |
| ---: | ---: | --- | ---: | ---: | ---: | --- |
| 16,002 | 131,072 | 224, chosen from the context | 173.4 s | 38.64 tokens/s | 9.44 | `RECEIPT.json` of the 16K leg (4f87949b) |
| 24,002 | 131,072 | 224, chosen from the context | 264.0 s | 36.34 tokens/s | 12.34 | `RECEIPT.json` of the 24K leg (96ef128e) |
| 16,002 | 24,576 | all 256 resident (expert cache off) | 170.6 s | 40.02 tokens/s | not applicable | `RECEIPT.json` of the resident leg (4e756e66) |

The 16K prompt gives identical text with 224 slots and with all experts resident (sha256 prefix 7e47dc5e). The peak video memory of the 131,072-token runs was 22,827 MiB (card) and 22,120 MiB (process). Prompt processing takes about 10.8 ms per token at 16,002 tokens and 11.0 ms at 24,002 and grows about linearly so far; ESTIMATE by extending that line, not measured: a full 128K prompt would take over 20 minutes. Moving the small per-step copies to a pinned stage and the layer-major prefill (its quality check failed; it stays off until it passes the registered gates above) are the levers for prompt speed.

Lazy block return (`FIFI_FLINT_CUDA_ALLOC_LAZY_RETURN`, 30 Oct 2026): with the switch on, a returned GPU block goes idle without a device-wide synchronize while the engine owns one stream. A/B on an otherwise idle host (load average 1.1 to 1.7 over all boots): default config, 131,072-token context, `f16` profile, 222 expert slots chosen from the context, no prefix reuse, the same real 4,000-token prompt, 128 greedy tokens, a fresh boot per arm, one warm-up boot discarded, then 8 timed boots in the order off, on, on, off, on, off, off, on. First token and decode are the server's telemetry.

| Arm | Boots | Decode | First token | Receipts |
| --- | ---: | ---: | ---: | --- |
| Switch off (default) | 4 | 39.82 tokens/s (39.25, 39.97, 40.03, 40.03) | 42.28 s | `RECEIPT.json` of each boot: 3264bcbd, 6110da75, 9679f6c0, 5ababa4e |
| `FIFI_FLINT_CUDA_ALLOC_LAZY_RETURN=1` | 4 | 40.90 tokens/s (40.92, 40.95, 40.87, 40.85) | 41.98 s | `RECEIPT.json` of each boot: 6cb9e81a, 8bf279da, f6737a87, 0ff56aff |

Decode is 2.71% faster with the switch on; first token is 0.7% shorter, which is within boot-to-boot noise. The text is identical in all nine boots (sha256 prefix c935d745) and the card peak is 22,799 MiB in all of them. The host load file is 6dfd5572 and the server binary is `enginesrv-lazy2` (ced99403). The gain is below the 3% set for switching it on by default, so the switch stays off; `FIFI_FLINT_CUDA_ALLOC_LAZY_RETURN=1` plus a restart turns it on. Earlier runs on a busy host showed +3.5% and are not used.

## Flint on Apple Silicon (M4 Max)

32,768 tokens of context by default. Prefix reuse is implemented and off by default. Setting (not a measurement): a 64 MiB freed-buffer cache for the resident decode loop is on by default since `8b6ecaf7c` (`FIFI_FLINT_MLX_CACHE_LIMIT=0` restores the previous behaviour). Its speed effect has no receipt on file, so no figure is given; the Mac rows below predate it and do not describe the current default. Profile not recorded in the receipts below, which ran on main `cda2a183a` on 30 Oct 2026.

| Measure | Value | Conditions | Source |
| --- | --- | --- | --- |
| First token, 4K prompt | 16.87 to 16.95 s | cold, bare, 3 boots, 4,038-token prompt, from the server telemetry | `summary.json` (ca3e0341) |
| Decode speed after a 4K prompt | 39.2 to 41.0 tokens/s (short answers) | same runs; before the cache default | same file |
| Decode speed, 384-token continuation after a 4K prompt | 34.4 tokens/s | 1 boot; before the cache default | `decode32k-summary.json` (d63a202d) |
| First token, 32K prompt | 61.5 s | cold, bare, 31,647 tokens, 1 boot | the decode memo (25fa0fde) |
| Decode speed after a 32K prompt | 18.5 tokens/s | same run; before the cache default | same memo |

About 8 s of the cold 4K first token is one-time model load and weight import (6.4 s lazy load plus 1.7 s first-evaluation import, per the memo above; later requests do not pay it). Between 13 and 16% of decode time is the memory allocator, measured with a 1 ms CPU sampler at both 4K and 32K depth (same memo).

## Models without a current measurement here

Nemotron 3.5 Super on CUDA, Qwen3.8-Flash-Next and Qwen3.8-27B have no measurement on this page. See [Models](../reference/models.md) for what is validated.

## Estimates and simulations, not measurements

- Lookup (n-gram) drafting for Qwen3.6 on the 3090 is a host-side simulation, not a served measurement. Replaying recorded generations against the verify-window cost curve measured on 21 Oct 2026 (FP16, 8K and 32K context), drafts of 1 to 3 tokens reach 0.88 to 1.00 of solo decode speed, and the best high-confidence policy reaches 1.00 to 1.08. That is under the 15% gain that would justify building it. The cost of a 16-row window on the current build is not measured yet and decides the one open case (long drafts on tool-call and edit traffic). Source: the lookup-drafting study write-up (ecb0f307).
