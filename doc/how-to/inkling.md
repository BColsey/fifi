# Run Inkling 2 Mini on Apple Silicon

The experimental native text worker reads the original mixed NVFP4/MXFP8 safetensors release. Download the text checkpoint and its tokenizer to a local SSD.

```sh
hf download thinkingmachines/Inkling-2-Mini \
  --revision ed40e7da625f594fc9c4990f5f66be24873c22d7 \
  --include config.json model.safetensors.index.json model-00001-of-00001.safetensors \
    tokenizer.json tokenizer_config.json chat_template.jinja generation_config.json \
  --local-dir ~/models/inkling2-mini
```

From the engine directory, build the native MLX server and start it with a short context. Set `ENGINESRV_API_KEY` to your local server credential before launch.

```sh
cargo build --release --locked --features native-mlx --bin enginesrv
FIFI_INKLING2_MINI_MLX=1 FIFI_MLX_AUTO_BUDGET=1 \
  target/release/enginesrv serve --model ~/models/inkling2-mini \
  --context 128 --listen 127.0.0.1:18795
```

Use the model name from `GET /v1/models`. The worker supports one active sequence and processes prompt tokens in order. It keeps 48 independent sets of KV and convolution history for the 24 physical layers. The CPU evaluates dense projections and attention. Metal evaluates the eight selected experts at each visit. The planner reserves space for dense weights, sequence state and staging, then uses the remaining budget for expert weights. On a cache miss, fifi reads the selected expert ranges from the checkpoint. The operating system can serve those reads from its file cache.

`FIFI_MLX_WORKING_SET_MIB` sets an optional upper bound on the Metal working-set budget. For example, set it to `5120` to reproduce the 5 GiB budget on the 16 GiB M4 mini. `GET /v1/recent-requests` reports the measured prompt and generation timings.

For chat requests, disabling thinking sets the publisher's reasoning effort to 0.0. Inkling can still emit reasoning, so a short output limit can be reached before the first answer token. The server returns reasoning and answer text in their respective fields. [Publisher guidance](https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/skills/inkling/SKILL.md#thinking-effort-set-it-everywhere)

The GGUF inspector recognizes the mainline `inkling2_loop` and repository-root `inkling2` layouts. Both experimental text adapters need the pinned publisher tokenizer. Use a directory containing one text GGUF or pass the selected model file, including files with a `.gguf_file` suffix.

```sh
FIFI_INKLING2_MINI_MLX=1 FIFI_INKLING2_MINI_GGUF_MLX=1 FIFI_MLX_AUTO_BUDGET=1 \
  target/release/enginesrv serve --model /path/to/Inkling-2-Mini-UD-IQ2_XXS.gguf \
  --tokenizer-dir ~/models/inkling2-mini --context 192 --listen 127.0.0.1:18795
```

| Artifact | Inspect | Load and generate | HTTP |
| --- | --- | --- | --- |
| Publisher NVFP4/MXFP8 safetensors | Yes | Real checkpoint, Metal experts | Raw, chat, and multi-turn WebGUI |
| Mainline `inkling2_loop` GGUF | Yes | Real UD-IQ2_XXS, Metal experts | Explicit file-path raw completions and streamed chat verified at `71aa350c` |
| Repository-root `inkling2` GGUF | Yes | Real UD-IQ2_XXS, Metal experts | Explicit file-path raw completions and streamed chat verified at `71aa350c` |

At `71aa350c`, `enginesrv check` and `serve --print-config` selected the native Inkling family for both real GGUF files and the publisher safetensors under the 6 GiB ceiling. Both GGUF artifacts returned the reference three-token continuation with temperature zero in the earlier `8a6c6347` HTTP runs. The table covers text. Vision and DSpark have separate qualification.

## Check the serving contract

Use a dedicated server on loopback with the original checkpoint and the verified 192-token context. Start it with a JSON config of this form, substituting the absolute model path.

```json
{
  "listen": "127.0.0.1:18795",
  "api_key_env": "ENGINESRV_API_KEY",
  "prefix_reuse": false,
  "sample_trace": true,
  "engines": [{
    "id": "inkling",
    "family": "inkling2_mini",
    "model": "inkling",
    "model_dir": "/path/to/inkling2-mini",
    "tokenizer_dir": "/path/to/inkling2-mini",
    "context_limit": 192
  }]
}
```

Launch it with the native opt-in and budget from above, using `enginesrv --config /path/to/config.json`. To exercise the WebGUI, set `web_dir` at the top level to the built WebGUI directory. Open the loopback address in a browser, select Inkling in the model picker, and send “Name the capital of France. Answer in one word.” For a follow-up, send “Which country is that city in? Answer in one word.” The measured browser run displayed “Paris” and “France” using a 192-token context and a 144-token output limit. Run the suite from the repository root once the model is ready. Python 3.9 or later is sufficient.

```sh
python3 -B engine/tests/tools/check_inkling_serving.py \
  --base-url http://127.0.0.1:18795 --model inkling \
  --report /path/to/scratch/inkling-serving-evidence.json
```

To verify raw text generation on the running server, send the same greedy three-token request used by the suite.

```sh
curl --fail-with-body --silent --show-error http://127.0.0.1:18795/v1/completions \
  -H "Authorization: Bearer ${ENGINESRV_API_KEY}" \
  -H 'Content-Type: application/json' \
  -d '{"model":"inkling","prompt":"The capital of France is","max_tokens":3,"temperature":0}'
```

The verified response has a `choices[0].text` value of `" Paris.\n\n"`, a `length` finish reason, five prompt tokens, and three completion tokens.

The report preserves JSON responses and stream events, including reasoning. The test checks raw and chat output caps, streaming parity, the sampled EOS ID and its inclusion in completion usage, explicit multi-turn history, repeatability across requests, and reset after an observed client disconnect. The raw reference check re-encodes the first three matching tokens of the llama.cpp UD-IQ2_XXS and native outputs using the pinned tokenizer. The server exposes sampled token IDs on the chat route.

## Mac qualification TODO

Use the 16 GiB M4 Mac mini for hardware results. The publisher safetensors revision used for these checks is `ed40e7da625f594fc9c4990f5f66be24873c22d7`. The token and context limits below are part of the workload. A fresh server can still read pages from a warm OS file cache.

| Priority | Work | Evidence and remaining checks |
| --- | --- | --- |
| 1 | Verify native text serving | On `f45062d7`, raw HTTP and the WebGUI produced “Paris.” The browser used the model picker and received a completed SSE stream. On `4138d0df`, the follow-up supplied all three messages and displayed “France” with a 144-token output allowance. Both streams ended at EOS. The 68-token control reached its limit before visible answer text. All 12 HTTP cases passed at `8a6c6347`, including cancellation and the following independent raw request. |
| 2 | Qualify bounded prefill | `c651c820` adds opt-in visit-major prefill. Two-row prefill was bitwise equal to serial for all 200,058 logits at each of six prompt positions and a follow-up. On a repeated 62-token prompt producing eight identical tokens at 5 GiB, the early two-row run increased time to first token from 36.39 s to 44.44 s and attributed reads from 19.92 GB to 33.30 GB. At `8a6c6347`, four rows also matched every serial logit on real weights, including the first row after prefill. The idle-host mapped-weight comparison below measured a 6–9 second reduction in time to first token with four rows, with mixed physical I/O results. Two- and four-row parity also passed on the `71aa350c` release test build. |
| 3 | Reduce attributed disk reads | The five-prompt-token, twelve-output-token repeated 5 GiB control on `f45062d7` reproduced 1.65 decode tok/s, 9.604 GB of logical expert reads and 6.551 GB of process-attributed disk reads. A 1,536-visit trace reproduced all 3,909 cache misses under byte-weighted LRU. All 4,388 repeated expert selections within the same token’s core passes hit the cache. Across tokens, 3,991 of 6,632 repeated selections hit. Layers 23 and 22 accounted for the most logical reads. Adjacent expert coalescing saved 312 read ranges on the first request and 282 on its repeat, measured against a one-range-per-expert-tensor replay with identical payload bytes. Frequency-based replay increased logical payload reads by 3.7%. The idle-host sustained comparison measured about 34% less process-attributed disk I/O with mapped weights and file-cache retention. Keep mapped storage opt-in. |
| 4 | Measure placement | On the safetensors short repeat, 11 decode steps spent 1.68 s in CPU attention, 3.90 s in expert staging including 2.80 s of demand reads, and 1.11 s in synchronized Metal readback. Real-weight one-row QKV tests favored four CPU workers over the widened Metal path. The short end-to-end safetensors run showed a small decode gain with four workers. The GGUF experiments below test attention workers and grouped readback separately. |
| 5 | Verify scaling | At a 192-token context, preflight rejected all three artifacts with a 1 GiB ceiling. The safetensors plan required 3,471,813,632 bytes, and each GGUF plan required 1,206,322,432 bytes. The mini applied the 5 GiB, 6 GiB and automatic ceilings in the idle-host run below. Automatic sizing incurred swap-outs on the first request and hit every expert on the repeat. The larger-device planner checks use simulated budgets. |
| 6 | Qualify each GGUF layout | `6df5ccc4` adds mainline and fork tensor adapters and a CPU reference path. At `4138d0df`, both complete IQ2_XXS artifacts selected token 152717 through the GPU decoder. The fork’s 200,058-logit vector had cosine similarity 0.999983 to the pinned Unsloth runtime and matched its top-ten set. Both layouts passed repeated raw HTTP at `8a6c6347`. The fork’s full-logit comparison at that revision matched all ten leading reference IDs with cosine similarity 0.999974. At `71aa350c`, all six Metal formats matched their CPU block references, and both full checkpoints selected the reference token. The fork matched all ten leading external-reference IDs. Both explicit file paths passed repeated raw completions and streamed chat at `71aa350c`. Measure the opt-in grouped projections and CPU workers. |

The comparison controls are opt-in. `FIFI_INKLING_PREFILL_ROWS=1` selects serial prefill. Larger values request up to 16 rows subject to admission. `FIFI_INKLING_EXPERT_TIMING=1` emits per-request cache totals and load, prefill and decode records. The prefill record includes computing logits for the final prompt token. With either expert timing or `FIFI_INKLING_COMPONENT_TIMING=1`, the phase log includes logical dense positional reads and logical mapped spans. Compare physical I/O using the separate process-attributed disk byte counter. `FIFI_INKLING_ROUTE_TRACE=1` emits each visit’s physical layer, selected and missing experts, cache capacity and read time. Use a separate trace run when measuring throughput.


### Initial 5 GiB storage comparison

These repeated requests used `4138d0df`, the publisher safetensors, a 128-token context, five input tokens and twelve greedy output tokens on the 16 GiB M4 mini. All four runs returned identical text and recorded zero system swap-outs. Each configuration ran in a fresh process with two identical requests. The OS page cache was uncontrolled.

| Dense weights | Packed expert retention | TTFT (s) | Decode (tok/s) | Wall (s) | Expert reads (GB) | Process-attributed disk reads (GB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Private | LRU | 3.41 | 1.52 | 10.66 | 9.60 | 6.70 |
| Mapped | LRU | 4.74 | 1.41 | 12.52 | 9.60 | 7.98 |
| Private | File cache only | 3.85 | 1.48 | 11.28 | 29.43 | 6.17 |
| Mapped | File cache only | 3.60 | 1.56 | 10.65 | 29.43 | 5.78 |

Set `FIFI_INKLING_MMAP_DENSE=1` to map safetensor dense weights and `FIFI_INKLING_FILE_CACHE_ONLY=1` to leave expert retention to macOS. A file-cache hit still contributes to the engine’s logical reads. File-backed pages can also be reclaimed under memory pressure. The fourth row merits further testing across prompt lengths and working-set ceilings.

### Balanced 5 GiB storage comparison

The `8a6c6347` server processed the same five prompt IDs and twelve greedy output tokens in four fresh processes. Each process served the request twice. The order was mapped/file cache, private/LRU, private/LRU, mapped/file cache. The following medians use two samples per cell. All eight outputs matched. The OS page cache was uncontrolled. Process-attributed disk bytes include disk I/O from mapped page faults that macOS attributes to the process. Process footprint excludes reclaimable file-backed pages.

| Request | Dense and expert retention | TTFT (s) | Prefill (tok/s) | Decode (tok/s) | Wall (s) | Logical expert reads (GB) | Attributed disk reads (GB) | Peak process footprint (GB) | Swap-out pages by run |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| First | Private and LRU | 7.64 | 0.66 | 0.84 | 20.70 | 9.99 | 10.97 | 4.85 | 0, 0 |
| First | Mapped and file cache | 7.92 | 0.64 | 1.09 | 18.05 | 29.43 | 9.72 | 0.47 | 310, 0 |
| Repeat | Private and LRU | 5.69 | 0.88 | 0.94 | 17.56 | 9.60 | 8.02 | 4.85 | 0, 0 |
| Repeat | Mapped and file cache | 4.99 | 1.00 | 0.99 | 16.14 | 29.43 | 7.76 | 0.47 | 0, 0 |

Mapped weights with file-cache-only experts reduced attributed disk reads by 11.5% on the first request and 3.1% on the repeat. The first request’s median TTFT rose by 0.28 s. One mapped run recorded system-wide swap-out activity. The OS cache and swap state differed from the initial experiment, and the absolute throughput varied across runs.

### Longer prompt at 5 GiB

The same `8a6c6347` binary processed 62 input and eight greedy output tokens with one prefill row at a time. The two processes ran mapped first and private second. All four outputs matched. The OS page cache was uncontrolled.

| Request | Dense and expert retention | TTFT (s) | Prefill (tok/s) | Decode (tok/s) | Wall (s) | Logical expert reads (GB) | Attributed disk reads (GB) | Peak process footprint (GB) | Swap-out pages |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| First | Private and LRU | 49.29 | 1.26 | 1.12 | 55.52 | 48.60 | 32.02 | 4.96 | 0 |
| First | Mapped and file cache | 53.63 | 1.16 | 1.11 | 59.96 | 126.92 | 22.39 | 0.46 | 0 |
| Repeat | Private and LRU | 65.74 | 0.94 | 0.36 | 85.28 | 48.38 | 42.80 | 4.97 | 37,250 |
| Repeat | Mapped and file cache | 61.27 | 1.01 | 1.00 | 68.30 | 126.92 | 32.15 | 0.46 | 0 |

Attributed disk reads fell by 30.1% and 24.9%. The first request’s time to first token increased by 4.34 s. The private-cache repeat coincided with 37,250 system-wide swap-out pages, which limits the throughput comparison. macOS can reclaim the mapped pages, and their total residency is outside the process footprint column.

### Idle-host four-row prefill at 5 GiB

The `8a6c6347` binary compared one row with four rows for the same 62 input and eight greedy output tokens. Both used mapped dense weights and file-cache-only expert retention. Each ran twice in its own server process, with the one-row process first. All four outputs matched. No competing build was detected. The OS page cache was uncontrolled.

| Request | Prefill rows | TTFT (s) | Prefill (tok/s) | Decode (tok/s) | Wall (s) | Logical expert reads (GB) | Attributed disk reads (GB) | Peak process footprint (GB) | Swap-out pages |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| First | 1 | 30.35 | 2.04 | 2.24 | 33.48 | 126.92 | 7.78 | 0.39 | 0 |
| First | 4 | 20.95 | 2.96 | 2.17 | 24.19 | 85.00 | 5.93 | 0.50 | 0 |
| Repeat | 1 | 27.09 | 2.29 | 2.22 | 30.24 | 126.92 | 5.54 | 0.39 | 0 |
| Repeat | 4 | 20.91 | 2.97 | 2.15 | 24.16 | 85.00 | 5.98 | 0.50 | 0 |

Four-row prefill reduced the first request’s time to first token by 9.40 s and the repeat’s by 6.18 s. It reduced logical expert payload reads during prefill by 36.8%, with traffic during decode unchanged. Attributed disk reads fell by 23.7% on the first request and rose by 7.8% on the repeat. Receipts `idle-long-file-only.json` and `idle-long-window4.json` include per-phase read ranges and byte counts.

### Sustained decode at 5 GiB

The `8a6c6347` server processed 14 prompt and 64 greedy output tokens. Each policy served the same prompt twice in one process, with the mapped process first. All four responses returned identical text about the history of Paris and reached the 64-token limit. The OS page cache was uncontrolled. A local iOS CI worker started during the first request, and Xcode began building during the first private-cache request. These measurements characterize the contention case.

| Request | Dense and expert retention | TTFT (s) | Prefill (tok/s) | Decode (tok/s) | Wall (s) | Logical expert reads (GB) | Attributed disk reads (GB) | Peak process footprint (GB) | Swap-out pages |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| First | Private and LRU | 15.74 | 0.89 | 0.66 | 111.59 | 49.65 | 46.23 | 5.03 | 0 |
| First | Mapped and file cache | 29.07 | 0.48 | 0.60 | 133.94 | 141.63 | 67.18 | 0.46 | 0 |
| Repeat | Private and LRU | 20.18 | 0.69 | 0.55 | 135.79 | 49.51 | 46.39 | 5.03 | 5,312 |
| Repeat | Mapped and file cache | 14.90 | 0.94 | 0.95 | 81.27 | 141.63 | 38.12 | 0.46 | 0 |

Mapped weights with file-cache-only retention increased first-request disk reads by 45.3% and reduced repeat reads by 17.8%. The private repeat overlapped with system swap-out activity.

### Idle-host sustained decode at 5 GiB

The exact `8a6c6347` binary repeated the sustained comparison with the private-cache process first. No competing build was detected during either process. The OS page cache was uncontrolled. All four responses matched and reached the 64-token limit.

| Request | Dense and expert retention | TTFT (s) | Prefill (tok/s) | Decode (tok/s) | Wall (s) | Logical expert reads (GB) | Attributed disk reads (GB) | Peak process footprint (GB) | Swap-out pages |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| First | Private and LRU | 10.31 | 1.36 | 1.55 | 50.91 | 49.65 | 35.44 | 5.03 | 0 |
| First | Mapped and file cache | 11.31 | 1.24 | 1.82 | 45.90 | 141.63 | 23.46 | 0.39 | 0 |
| Repeat | Private and LRU | 9.54 | 1.47 | 1.55 | 50.06 | 49.51 | 33.94 | 5.03 | 0 |
| Repeat | Mapped and file cache | 8.57 | 1.63 | 1.76 | 44.41 | 141.63 | 22.54 | 0.39 | 0 |

Process-attributed disk reads fell by 33.8% on the first request and 33.6% on its repeat. Decode throughput rose by 17.4% and 13.5%. The first request took one second longer to produce its first token and completed 5.01 seconds sooner. The footprint metric excludes reclaimable mapped pages. Receipts `idle-sustain-private.json` and `idle-sustain-file-only.json` include the separate prefill and decode counters.

### Idle-host budget and CPU worker comparison

The 16 GiB M4 mini reported a Metal recommendation of 12,713,115,648 bytes. For the 128-token context, the shared planner reserved 1 GiB for memory pressure and accounted for 2,385,439,744 bytes of dense weights, state and staging. A configured 5 GiB budget left 1,909,527,552 bytes for packed experts. At 6 GiB the cache limit was 2,983,269,376 bytes. Automatic sizing admitted 9,253,934,080 bytes for experts.

The `8a6c6347` runs used private dense weights and LRU with the same five prompt tokens and twelve greedy output tokens. Each fresh process served two requests. All eight outputs matched. The OS page cache was uncontrolled.

| Ceiling and placement | Request | TTFT (s) | Prefill (tok/s) | Decode (tok/s) | Wall (s) | Logical expert reads (GB) | Attributed disk reads (GB) | Peak process footprint (GB) | Swap-out pages |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 GiB | First | 4.77 | 1.05 | 1.82 | 10.82 | 9.99 | 9.60 | 4.82 | 0 |
| 5 GiB | Repeat | 3.01 | 1.66 | 2.19 | 8.04 | 9.60 | 4.77 | 4.85 | 0 |
| 5 GiB, four CPU workers | First | 4.17 | 1.20 | 1.87 | 10.06 | 9.99 | 8.62 | 4.85 | 0 |
| 5 GiB, four CPU workers | Repeat | 2.81 | 1.78 | 2.22 | 7.77 | 9.60 | 5.10 | 4.85 | 0 |
| 6 GiB | First | 5.57 | 0.90 | 1.55 | 12.66 | 8.10 | 8.44 | 5.95 | 0 |
| 6 GiB | Repeat | 3.51 | 1.42 | 1.66 | 10.15 | 7.41 | 6.93 | 5.95 | 0 |
| Automatic | First | 5.38 | 0.93 | 1.02 | 16.20 | 6.35 | 8.26 | 9.11 | 75,980 |
| Automatic | Repeat | 2.83 | 1.77 | 2.72 | 6.87 | 0.00 | 0.00 | 9.11 | 0 |

The automatic run retained 6,347,464,200 bytes of expert weights and hit every routed expert on the repeat. The first request coincided with 75,980 system-wide swap-out pages. The competing-build run recorded 170,004 swap-out pages and 3.24 GB of process-attributed reads on the automatic repeat. The idle run still shows a first-request memory pressure cost, even though the automatic repeat performed no disk reads. The four-worker short run used the same logical read bytes as its 5 GiB control, and repeat wall time decreased by 0.27 s. Repetition across fresh processes is needed to separate placement effects from OS cache variability. Metal’s recommendation describes GPU resource allocations. [Apple Metal working-set documentation](https://developer.apple.com/documentation/metal/mtldevice/recommendedmaxworkingsetsize)

### Short storage control at `71aa350c`

Two fresh processes ran the same five prompt tokens and generated the same twelve greedy output tokens twice at 5 GiB. The mapped process ran first. An initial private process was stopped while a build occupied the mini, so the private row uses the retry after the device became idle. No competing builds were detected during either reported run. The runs were separated by about fifteen minutes, and the OS page cache was uncontrolled.

| Request | Dense and expert retention | TTFT (s) | Prefill (tok/s) | Decode (tok/s) | Wall (s) | Logical expert reads (GB) | Attributed disk reads (GB) | Peak process footprint (GB) | Swap-out pages |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| First | Private and LRU | 4.56 | 1.10 | 1.57 | 11.56 | 9.99 | 6.84 | 4.85 | 0 |
| First | Mapped and file cache | 6.97 | 0.72 | 1.73 | 13.31 | 29.43 | 9.15 | 0.47 | 0 |
| Repeat | Private and LRU | 3.00 | 1.67 | 1.91 | 8.75 | 9.60 | 4.82 | 4.85 | 0 |
| Repeat | Mapped and file cache | 3.17 | 1.58 | 1.76 | 9.43 | 29.43 | 5.28 | 0.47 | 0 |

The mapped repeat read 0.46 GB more from disk and took 0.68 s longer. Its prefill and decode accessed 54.79 GB of logical mapped dense spans, which can overlap the same file pages many times. The results reinforce the need to measure physical I/O for each workload before choosing a policy.

### Packed GGUF first-logit probes at 6 GiB

Both variants used the same real IQ2_XXS checkpoints, mapped dense projections, a 128-token context, and the same five prompt token IDs. Each execution stopped at the first output logit vector. The OS page cache was uncontrolled.

| Layout | Three packed formats at `8a6c6347` | Six packed formats at `71aa350c` | CPU attention projections at `71aa350c` | Metal expert readback at `71aa350c` |
| --- | ---: | ---: | ---: | ---: |
| Unsloth fork | 34.15 s | 11.26 s | 5.39 s | 2.64 s |
| Mainline | 33.90 s | 13.39 s | 6.02 s | 2.12 s |

Both selected token 152717. The fork logit vector had cosine similarity 0.999973 to the pinned Unsloth llama.cpp reference and matched the ordered top ten. The six-format test logged 3,840 packed projections and zero expanded expert projections on each layout. CPU attention projection took the largest measured share. Metal readback includes evaluating the submitted expert work.

### GGUF serving at 6 GiB

Under the ordinary explicit-file launch with `71aa350c`, both layouts returned `" Paris.\n\n"` twice. The streamed chat then displayed “Paris,” returned separate reasoning, and ended at EOS with `[DONE]`. The chat requests used 24 input tokens, a 192-token context, and the same settings.

| Layout | Raw repeat TTFT | Raw repeat decode | Chat TTFT | Chat decode | Chat wall time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unsloth fork | 7.16 s | 0.57 tok/s | 38.43 s | 0.55 tok/s | 139.26 s |
| Mainline | 6.96 s | 0.58 tok/s | 42.34 s | 0.53 tok/s | 145.64 s |

Both chats used 56 completion tokens including EOS. CPU attention projections took 72.87 s on the fork and 73.79 s on mainline across prefill and decode. Metal expert readback took 30.03 s and 30.69 s. Each three-token raw repeat recorded 8,192 process-attributed disk bytes and zero swap-out pages. The OS page cache was uncontrolled.

The pinned Unsloth llama.cpp reference `b12588-mix-c0c3de8` reported 4.50 prefill tok/s and 13.74 decode tok/s on the fork with 12 output tokens and full Metal offload. Its effective context was 256 tokens and it did not use fifi's 6 GiB budget. Those are reference-runtime observations.

### Cache and placement experiments

`FIFI_INKLING_GGUF_SINGLE_COPY_EXPERTS=1` uses the full already-admitted expert budget for packed Metal buffers when every stored expert projection has a direct packed kernel. Set it alongside `FIFI_INKLING_GGUF_PACKED_MLX=1`. The application retains one copy of each cached packed projection. The phase trace reports the two tier capacities and retained payload bytes. Use the same ceiling with and without the flag when comparing it.

`FIFI_MLX_ADAPT_EXPERT_CACHE=1` applies to the publisher safetensors host cache. The decoder halves its admitted expert retention limit after a macOS warning and clears retained experts after a critical event. It checks the event before staging a visit or when resetting the decoder. The reduced cap lasts until server restart. KV and staging allocations stay within the original plan. The SDK smoke check registered and reused the observer. Test the integrated runtime and capture naturally occurring pressure events before comparing physical I/O. [Apple Dispatch memory-pressure documentation](https://developer.apple.com/documentation/Dispatch/DISPATCH_SOURCE_TYPE_MEMORYPRESSURE)

`FIFI_INKLING_GGUF_BATCH_EXPERTS=1` groups eight gate/up projections and eight down projections on layers with two direct packed kernels. It issues one readback per group and retains the same CPU activation and output accumulation order. The shared planner reserves 48,365,568 additional bytes for the live and deferred groups. At `e9f74c2b`, the real fork test matched projected expert vectors and merged visit outputs bitwise across seven layers, including zero-retention and eviction cases. With warm retention, total grouped MLX import and readback synchronizations fell from 32 to four per visit. Combined packed, single-copy, grouped and four-worker probes on both real layouts each recorded 480 expert readbacks across 240 visits. Both selected token 152717. The fork matched the ordered top ten from the external reference with cosine similarity 0.999973.

`FIFI_INKLING_ATTN_CPU_WORKERS=4` also enables up to four workers for GGUF Q, K, V, and attention output projections. They handle contiguous output rows while preserving each row’s dequantization and accumulation order. Four workers add 8,974,336 bytes to the shared scratch reservation, including bounded positional-read buffers and worker stacks. At `e9f74c2b`, all 200,058 first-generation logits after a five-token prompt matched bitwise with one and four workers on each real file. Preflight verified the independent grouped-expert and worker charges on both files. With both optimizations enabled, each GGUF preflight required 1,263,707,392 bytes and rejected a 1 GiB ceiling. Matched request latency, CPU time and SSD byte measurements remain in progress.

### Matched Unsloth GGUF fork requests at 6 GiB

The exact `e9f74c2b` binary processed five prompt tokens and twelve greedy output tokens twice in each fresh process. It used a 192-token context, mapped dense weights and direct packed Metal kernels. The three trials each changed one setting. Controls ran before and after them. No competing build was detected. All ten outputs matched and reached the requested limit. The OS page cache was uncontrolled.

| Setting | Request | TTFT (s) | Prefill (tok/s) | Decode (tok/s) | Wall (s) | Logical expert reads (GB) | Attributed disk reads (MB) | Peak process footprint (GB) | Swap-out pages |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Control before | First | 7.78 | 0.64 | 0.53 | 28.39 | 3.10 | 1,365.01 | 4.51 | 0 |
| Single-copy cache | First | 7.65 | 0.65 | 0.57 | 26.93 | 3.10 | 0.12 | 3.37 | 0 |
| Grouped readback | First | 6.63 | 0.75 | 0.65 | 23.60 | 3.10 | 117.51 | 4.53 | 0 |
| Four CPU workers | First | 5.40 | 0.93 | 0.78 | 19.43 | 3.10 | 217.09 | 4.53 | 0 |
| Control after | First | 7.88 | 0.63 | 0.57 | 27.33 | 3.10 | 134.86 | 4.60 | 0 |
| Control before | Repeat | 6.93 | 0.72 | 0.58 | 25.77 | 0 | 0 | 4.51 | 0 |
| Single-copy cache | Repeat | 6.93 | 0.72 | 0.58 | 25.74 | 0 | 0 | 3.37 | 0 |
| Grouped readback | Repeat | 5.91 | 0.85 | 0.67 | 22.23 | 0 | 0.30 | 4.53 | 0 |
| Four CPU workers | Repeat | 4.45 | 1.12 | 0.83 | 17.77 | 0 | 1.32 | 4.59 | 0 |
| Control after | Repeat | 6.95 | 0.72 | 0.58 | 25.88 | 0 | 27.21 | 4.60 | 0 |

Four workers halved CPU attention projection time from about 14.3 s to 7.1 s on the repeat. Grouped execution reduced expert readbacks from 12,288 to 1,536 across the full request and reduced time in readback from about 5.58 s to 2.25 s. Single-copy retention lowered peak footprint by 1.14–1.24 GB while matching both controls’ repeat decode rate. The first requests read the same logical expert bytes. Physical traffic varied by more than 1.2 GB between the two first-request controls, so this run does not establish a physical I/O reduction from the three flags.

### Speculative read replay

The measured 5 GiB route trace spent 9.13 s in demand reads across 1,536 visits. Two historical predictors used the previous token’s route at the same visit or the latest route for the same physical layer. Both would issue zero speculative reads under a one- or two-visit lookahead because all their predicted experts were already cached when they would have issued. At a simulated 1 GiB expert cache, the better two-visit predictor by byte precision would read 16.63 GB ahead, including 11.08 GB unused at its target visit. This replay models logical payloads. The runtime retains demand-only scheduling.

## Reproduce a first and repeat benchmark

Run the driver from the repository root with the release server binary and a new output directory.

```sh
FIFI_INKLING_PREFILL_ROWS=1 python3 -B engine/tests/tools/bench_inkling.py \
  --server engine/target/release/enginesrv --model ~/models/inkling2-mini \
  --output-dir /path/to/scratch/inkling-5g --ceiling-mib 5120 --max-tokens 12
```

The driver starts one server on an OS-assigned loopback port, sends two identical greedy requests, and checks that both reach the requested output limit. It writes the server config, startup and phase log, and a JSON receipt with wall time, request timings, process-attributed disk bytes, and system swap-page deltas on macOS. Existing Inkling environment settings carry through to the server. Use `--ceiling-mib auto` to remove a configured ceiling, `--tokenizer-dir` for a separate pinned bundle, and `--commit`, `--build`, and `--checkpoint-id` to record artifact provenance. The OS page cache is uncontrolled across runs.
