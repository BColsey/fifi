# Benchmarks

These numbers show what fifi does on the hardware it is tested on.
Each table compares runs with the same weights on the same machine, measured in the same window on the same day.
Every fifi figure below was measured on 20 Nov 2026 on main as of that date.
Workload: greedy sampling, 256 generated tokens, fresh server boot per run, three runs per prompt size; ranges list all three runs.
Reference engines appear only where a same-window pair exists — llama.cpp on the RTX 3090 and MLX-LM on Apple Silicon. Hosts that cannot run a reference engine carry single-engine figures and say so.

Day-to-day noise is about 4%, so differences smaller than that are not meaningful.
"Prefill" is the server's own prefill interval; "decode" is its tokens-per-second counter. Where a reference engine reports differently (prompt-eval and eval intervals), the table names the column.

## Qwen3.6-35B-A3B, GGUF Q4_K_M, on an RTX 3090 (24 GB)

fifi: the default arm (nothing set; KV cache fp16, context 32,768). llama.cpp: `llama-server` 0.4.0-dev build 10934, commit `acecd5603`, CUDA build, `-c 32768 -ngl 99 --temp 0`, the same GGUF file and the same prompt files, four slots of 32,768 context. Same window, same card, same host. Figures are fifi `prefill_ms` vs llama.cpp `prompt_eval_ms` (both exclude generation), and each engine's own decode counter; each cell carries its three runs.

| Prompt tokens | fifi prefill ms | llama.cpp prefill ms | fifi decode tok/s | llama.cpp decode tok/s |
| ---: | --- | --- | --- | --- |
| 510 | 144.5 / 145.4 / 146.1 | 245.10 / 241.71 / 238.07 | 164.71 / 164.95 / 165.28 | 145.73 / 143.55 / 145.51 |
| 4,010 | 687.3 / 688.2 / 691.6 | 1,381.32 / 1,414.53 / 1,390.14 | 159.32 / 159.51 / 160.10 | 141.04 / 143.26 / 142.95 |
| 32,010 | 7,585.7 / 7,632.2 / 7,636.0 | 11,859.71 / 12,114.23 / 12,186.57 | 130.54 / 133.55 / 134.07 | 129.24 / 125.06 / 125.35 |

On this pair fifi is faster at every size: prefill by 1.6x to 2.0x, decode by 7 to 13 percent (medians 165.0 vs 145.5, 159.5 vs 143.0, 133.6 vs 125.4). A 128,010-token prompt is refused by fifi's default 32,768 context; llama.cpp was not run at that size.

## Qwen3.6-35B-A3B, MLX 4-bit, on Apple Silicon (36 GB)

fifi: the shipped 36 GB config (`engine/config/qwen36-mac-studio-36gb.json`) with paths swapped — see [Current status](status.md) for its figure table and switch list. The default arm refuses to start on this host (missing `cache_limit_bytes`), so the tuned arm is the only side that can pair.

MLX-LM 0.32.2 (Python 3.12 venv; the older 0.29.1 refuses this model type), the same model file, the same prompt strings, greedy sampling, 256 generated tokens, measured in-process (no HTTP layer), same window and host.

Per-phase, at 526 / 4,022 / 32,028 prompt tokens:

| Metric | fifi | MLX-LM |
| --- | --- | --- |
| Prefill tok/s | 1,098 / 1,246 / 928 | 1,449 / 1,661 / 1,221 |
| Decode tok/s | 86.9 / 78.8 / 66.8 | 117.0 / 115.3 / 95.4 |

**MLX-LM is faster at every size on this host: 24 to 32 percent faster prefill and 26 to 30 percent faster decode.** This is the headline gap for the MLX backend.

Caveats, stated: the MLX-LM figure is in-process (no HTTP, no server), while the fifi figure is the server's own counters; both are greedy on the same file. The comparison is honest about its layers but is not a pure like-for-like server pair.

For single-layer Q8 decode and dense-prefill timings on a 16 GB M4 Mac mini,
see the [M4 attention measurements](mac-m4-attention.md).

## Flint, MLX int4 21 GB fixture, on the same Apple Silicon host

No reference pair exists: MLX-LM cannot load this model type ("Model type flint not supported"), so this table is single-engine, measured for admission and ceiling rather than comparison. Default arm (which equals the tuned arm — nothing qualified to switch on for Flint on MLX):

| Prompt tokens | Prefill ms (median of 3) | Decode tok/s | Conditions |
| ---: | ---: | --- | --- |
| 526 | 424 | 52.34 / 52.73 / 52.75 | greedy, 256 tokens |
| 4,022 | 2,944 | 48.41 / 48.52 / 48.54 | same |
| 32,028 | 46,986 | 26.04 / 26.07 / 26.11 | fits under the env-fixed 32,768 ceiling; a 128K prompt refuses |

Peak server RSS 14.1 GiB. For the model-on-model comparison against Qwen3.6-MLX on the same host (32K prefill 36% slower, decode 61% slower), see [Current status](status.md).

## Qwen3.6-35B-A3B, GGUF Q4_K_M, on an AGX Xavier board

No reference engine runs on this board (its software set does not include one and its configuration is frozen), so these are single-engine figures, useful for admission and thermal planning rather than comparison. Same weights file as the 3090 tables.

| Arm | Prompt tokens | Prefill | Decode tok/s | Conditions |
| --- | ---: | --- | --- | --- |
| Default | 486 | 2,911.8–4,296.4 ms | 20.57–20.69 | two runs, peak 54.5 °C |
| Default | 3,993 | 35,755–37,128 ms | 19.92–19.96 | three runs, peaks 60.5–67.0 °C |
| Default | 31,650 | 28.0 and 29.5 min | 14.48 / 15.38 | two completed runs, run before the thermal guard existed (the board reached 91 °C in that period); three later runs hit the 88 °C thermal guard during prefill |
| Tuned (KV q8, context 16,384, capture and prefix reuse) | 486 | 2,788.3–3,242.0 ms | 20.65–20.75 | three runs, temperature not recorded |
| Tuned | 3,993 | 36,864.9–43,020.3 ms | 19.94–20.04 | three runs, temperature not recorded |
| Tuned | 15,283 (max admitted) | 401,369.9 ms (~6.7 min) | 18.29 | one completed run, peak 87.0 °C; two later runs hit the guard |

The board holds ~20 tok/s decode at every admitted size; its limits are prefill time and the 88 °C thermal guard. For the same weights on an RTX 3090, see the first table on this page.
