# Current status

**Goal: 128K to 256K tokens of context for every model, with EXL3 as the primary weight format.** The context rows show the measured admission limits and boot results.

This page records measurements by model and backend.
It holds measured facts only. Every row gives the date, the conditions and the receipt it comes from, and anything that is an estimate says so.
Numbers move with every merged change, so the date on each row matters more than the order of the rows.

Every row below comes from the same-day measurement sweep of 20 Nov 2026 on the frozen tip `47ad58e02` (server binary sha8 `54a0ab57` on the CUDA hosts, `500406ee` on Apple Silicon). Figures from earlier sweeps that this sweep did not reproduce were removed rather than kept; where a model was not measurable that day, the row says so instead of carrying an old number. Workload throughout: greedy sampling, 256 generated tokens, fresh server boot per run, three runs per prompt size (ranges list all three; medians in parentheses where useful).

How to read the rows:

- **Default arm** means the server was started with nothing set: a config that names only the engine and model files, no tuning switches. What a user gets who sets nothing.
- **Tuned arm** means the shipped profile for that host (named in each section) with only paths and the listen address swapped. Nothing hand-built.
- **Prefill / first token** is the server's own `prefill_ms` / `time_to_first_token_ms` from the request log. **Decode** is `decode_tokens_per_second`. A row that was measured without temperature recording says so.
- **Refusal rows** quote the exact error the server returned. A refusal is a result, not a missing row.
- **Units**: GB is 10^9 bytes; MiB and GiB are binary (2^20 and 2^30 bytes). Rows that quote nvidia-smi use MiB.
- Receipts are kept in the project's measurement archive, not in this repository. Each source names the receipt kind and the date; the sweep's per-run server logs are the underlying files.
- Compare rows only when their conditions match; for comparisons against other engines, see [Benchmarks](benchmarks.md).

## Qwen3.6-35B-A3B, GGUF Q4_K_M, on an RTX 3090 (24 GB, CUDA)

Model file served in place (one GGUF plus its tokenizer; 20,419,565,568 bytes of weights). See [Run Qwen3.6](../how-to/qwen36.md).

Default arm, load line: gguf weights resident 20,690,501,632 B, 3,627,941,888 B free after load, context limit 32,768, KV cache fp16, graph capture on.

| Prompt tokens | Prefill ms | Decode tok/s | Date | Conditions |
| ---: | --- | --- | --- | --- |
| 510 | 144.5 / 145.4 / 146.1 | 165.28 / 164.95 / 164.71 | 20 Nov 2026 | three fresh boots; server request log |
| 4,010 | 691.6 / 687.3 / 688.2 | 159.51 / 159.32 / 160.10 | 20 Nov 2026 | same |
| 32,010 | 7,632.2 / 7,585.7 / 7,636.0 | 133.55 / 134.07 / 130.54 | 20 Nov 2026 | same |
| 128,010 | refused | refused | 20 Nov 2026 | HTTP 409 `context_limit`: "prompt is too long: 128010 tokens > 32767 maximum" |

Tuned arm: the shipped profile `engine/config/qwen36-cuda-3090-kv8-prefix.json` (prefix reuse on, context limit 180,224, `kv_dtype` q8, capture on, activation fp32). Load line: same resident weights, 3,578,134,528 B free after load, KV cache int8g32.

| Prompt tokens | Prefill ms | Decode tok/s | Date | Conditions |
| ---: | --- | --- | --- | --- |
| 510 | 164.7 / 150.6 / 145.8 | 153.58 / 158.87 / 161.01 | 20 Nov 2026 | three fresh boots; server request log |
| 4,010 | 943.9 / 941.1 / 945.0 | 161.07 / 161.17 / 161.06 | 20 Nov 2026 | same |
| 32,010 | 8,697.6 / 9,073.4 / 9,116.9 | 136.23 / 131.83 / 129.37 | 20 Nov 2026 | same |
| 128,010 | 54,824.9 / 54,598.0 / 55,390.3 | 85.42 / 86.39 / 83.50 | 20 Nov 2026 | the 128K file fits the tuned cap; prefill runs at ~2,330 tok/s |

Prefix reuse on the default arm: the save store fills after a 4,010-token turn (2 entries, 297,594,880 B saved), but a stateless second request over the same text does not reuse it (`prefix_hit miss`, reused 0). First-turn ttft 709.3 ms; second-turn 835.6 ms (4,518 tokens). Reuse across raw stateless completions is not a win; it needs a session that carries the boundary.

Memory caps (default arm, read-only-mounted container, `--memory` per start, one start each): 6, 8, 12 and 20 GiB all **serve**; the tier line is identical at every cap (weights stream to the device, 20.7 GiB resident there; free-after-load 3,474,784,256 B; context 32,768; KV fp16). The 500-token request wall shrinks as the cap grows: 29.0 s at 6 GiB, 22.5 s at 8, 14.0 s at 12, 7.9 s at 20.

Small card: with a holder leaving 5,836 MiB of device memory free, the default start refuses with "engine q36 failed: engine load failed: cuda admission: need 22153406976 bytes (tensors 20408576512, KV cache 671088640, work 0, headroom 1073741824), device has 5805441024 free". Qwen3.6 does not spill weights to host memory on a small card; a card that cannot hold the tensors plus admission does not serve.

## Qwen3.8 Flash-Next, EXL3 3.05 bpw, on the same RTX 3090

Default arm: context limit 4,096 (the EXL3 default), server request log, three runs per size.

| Prompt tokens | Prefill ms | Decode tok/s | Date | Conditions |
| ---: | --- | --- | --- | --- |
| 571 / 572 | 8,207.0 / 8,621.3 | 29.16 / 29.94 | 20 Nov 2026 | two of the three runs carried in the sweep summary; greedy, 256 generated |
| 4,071 / 4,072 | 61,769.1 / 61,094.8 / 61,136.3 | 23.74 / 22.73 / 23.36 | 20 Nov 2026 | the largest admitted 4,064-of-4,096 run decodes at 26.57 tok/s |

Tuned arm (ring budget 1,155,302,400, EXL3 rows max 2,048, batch 1): 500-token prefill 7,838.5 / 7,539.9 / 7,602.9 ms, decode 27.57 / 28.46 / 28.49 tok/s; 4,072-token prefill 56,986.2 ms on the carried run, decode 22.48 tok/s.

Prefix reuse, both arms, a two-turn conversation whose second turn repeats the first ~3,000 tokens: the second turn reuses them — reused 2,895 (default) and 3,216 (tuned) tokens, ttft 5.1 s and 4.7 s against 44.0 s and 39.6 s cold for that prompt. Reuse is the one tuned setting that pays on this model.

Memory caps (default arm): 6, 8 and 12 GiB refuse before the arm with the fixed-overhead text (the EXL3 banks need 14.18 GiB before any prompt); 20 GiB serves on the disk tier (no pinned store): ready in 20.1 s, a 67-token prompt prefills cold in 6,086 ms (11.7 tok/s decode) and warm in 173–200 ms (20.9–22.9 tok/s).

## Qwen3.6-35B-A3B, MLX 4-bit, on Apple Silicon (36 GB)

Model file: the 19G MLX 4-bit pack.

Default arm: **refused at startup**. Exact text: "enginesrv: serve --model: this build cannot serve ...: settings: the engine would refuse these settings: engine config: cache_limit_bytes is missing and Q36_CACHE_LIMIT is not set". A user who sets nothing gets no server on this host class; this is the top item a fix should remove.

Tuned arm: the shipped config `engine/config/qwen36-mac-studio-36gb.json` verbatim (cache limit 4 GiB, context 262,144, session cap 4 GiB, prefill budget 30 GiB, reserve 8,286,699,520, bf16+q8, compile on) with model dir and listen swapped. Timings from `/v1/recent-requests`, three runs per size:

| Prompt tokens | Prefill ms (median) | Decode tok/s | Date | Conditions |
| ---: | ---: | --- | --- | --- |
| 526 | 480 (479 / 480 / 892; the 892 first hit pays compile tracing) | 86.82 / 86.93 / 86.90 | 20 Nov 2026 | greedy, 256 tokens |
| 4,022 | 3,228 (3,241 / 3,226 / 3,228) | 78.76 / 78.79 / 78.81 | 20 Nov 2026 | same |
| 32,028 | 34,532 (34,543 / 34,532 / 34,514) | 66.77 / 66.94 / 66.75 | 20 Nov 2026 | same |

Peak server RSS 15.7 GiB (unified memory; process RSS is the host+device proxy). Load to ready ~5.5 min cold including MLX compile tracing. A third admission of the 4,022-token text shows no reuse win (3,243 ms median): raw completions mark no prefix boundary, as the API contract documents. Against MLX-LM on the same file, see [Benchmarks](benchmarks.md).

Dense Qwen3.8-27B on this host: not servable. Both local copies are EXL3 weights, and the server refuses at startup with "engine config: weight_format exl3 is CUDA-only". No MLX-format 27B pack exists on the host; the row needs one before it can hold a figure.

## Flint, MLX int4 21 GB fixture, on the same Apple Silicon host

Default arm equals the tuned arm: nothing qualified to switch on for Flint on MLX, and the MLX path needs no Flint config object (the object itself selects the CUDA pilot). Config: id flint, family flint, model and tokenizer dirs only. Server-side timings, three runs per size:

| Prompt tokens | Prefill ms (median) | Decode tok/s | Date | Conditions |
| ---: | ---: | --- | --- | --- |
| 526 | 424 (415.4 / 424.0; first 2,262 pays cold) | 52.34 / 52.73 / 52.75 | 20 Nov 2026 | greedy, 256 tokens |
| 4,022 | 2,944 (2,943.7 / 2,899.0 / 3,240.4) | 48.41 / 48.52 / 48.54 | 20 Nov 2026 | same |
| 32,028 | 46,986 (46,853 / 46,986 / 46,990) | 26.04 / 26.07 / 26.11 | 20 Nov 2026 | same |

Peak server RSS 14.1 GiB. Load to ready ~1 min. The context ceiling for this arm is the env-fixed 32,768; the 32,028+256 prompt fits under it, a 128K prompt would refuse. A third admission of the 4,022-token text again shows no reuse win (raw route, as contracted). On the same host, 32K prefill is 36% slower than Qwen3.6-MLX (46.99 s vs 34.53 s) and decode is 61% slower (26.1 vs 66.8 tok/s); the fixture is the parked family, measured as-is.

## Qwen3.6-35B-A3B, GGUF Q4_K_M, on an AGX Xavier board

Same weights file as the 3090 rows. No reference engine runs on this board, so every row is single-engine. The board's thermal guard trips at 88 °C; runs that hit it during prefill are recorded as guard aborts, and completed runs carry their peak temperature.

Default arm:

| Prompt tokens | Prefill ms | Decode tok/s | Peak temp | Date | Conditions |
| ---: | --- | --- | --- | --- | --- |
| 486 | 4,296.4 / 2,911.8 | 20.57 / 20.69 | 54.5 °C | 20 Nov 2026 | two runs, temperature recorded |
| 3,993 | 35,755–37,128 | 19.92–19.96 | 60.5–67.0 °C | 20 Nov 2026 | three runs, temperature recorded |
| 31,650 | 1,681,370 / 1,767,007 | 14.48 / 15.38 | not recorded | 20 Nov 2026 | two completed runs (~28.0 and ~29.5 min prefill each); three later runs aborted at the 88 °C guard during the ~13-minute prefill |

Tuned arm (KV q8, context 16,384, capture and prefix reuse on; temperature not recorded on these runs):

| Prompt tokens | Prefill ms | Decode tok/s | Date | Conditions |
| ---: | --- | --- | --- | --- |
| 486 | 3,242.0 / 2,802.8 / 2,788.3 | 20.75 / 20.72 / 20.65 | 20 Nov 2026 | three runs |
| 3,993 | 43,020.3 / 41,713.5 / 36,864.9 | 20.04 / 19.94 / 20.02 | 20 Nov 2026 | three runs |
| 15,283 (max admitted) | 401,369.9 | 18.29 | 20 Nov 2026 | one completed run, peak 87.0 °C; two later runs aborted at the 88.0 °C guard |

A 31,650-token prompt over the tuned 16,384 cap refuses with the same `context_limit` 409 as the 3090 default arm. The board sustains ~20 tok/s decode at every admitted size; its constraint is prefill time and the thermal guard, not decode.

## Models without a current measurement here

- **Qwen3.6 EXL3 4.00 bpw on the 3090**: no pack on the host during the 20 Nov 2026 sweep; the rows from the October sweeps were removed with this rewrite. The format is merged and selected with `"weight_format": "exl3"`; a same-day figure needs the pack present.
- **Qwen3.6 EXL3 2.08 bpw on an RTX 4050 laptop**: the pack's `tokenizer.json` hash does not match the pinned revision, and the server refuses to start. The row needs a repacked or re-pinned tokenizer file.
- **Flint on the 3090**: refused — "engine flint failed to start: tokenizer.json is missing" (no Flint pack on that host).
- **Bolt e4b on the 3090**: refused — "serve: 4 .gguf files in the model directory; exactly one expected" (the store keeps four quantizations of this model in one directory and the loader accepts exactly one file).
- **Nemotron Super 35B on the 3090**: refused — the loader for this family is safetensors-based and the store holds only a GGUF conversion: "config.json: No such file" then "open shard: No such file".

## Estimates and simulations, not measurements

- Lookup (n-gram) drafting for Qwen3.6 on the 3090 is a host-side simulation, not a served measurement. Replaying recorded generations against the verify-window cost curve measured on 21 Oct 2026 (FP16, 8K and 32K context), drafts of 1 to 3 tokens reach 0.88 to 1.00 of solo decode speed, and the best high-confidence policy reaches 1.00 to 1.08. That is under the 15% gain that would justify building it. The cost of a 16-row window on the current build is not measured yet and decides the one open case (long drafts on tool-call and edit traffic). Source: the lookup-drafting study write-up (ecb0f307).
