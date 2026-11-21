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
| Mainline `inkling2_loop` GGUF | Yes | Real UD-IQ2_XXS, Metal experts | Raw completion verified at `8a6c6347` |
| Repository-root `inkling2` GGUF | Yes | Real UD-IQ2_XXS, Metal experts | Raw completion verified at `8a6c6347` |

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
| 2 | Qualify bounded prefill | `c651c820` adds opt-in visit-major prefill. Two-row prefill was bitwise equal to serial for all 200,058 logits at each of six prompt positions and a follow-up. On a repeated 62-token prompt producing eight identical tokens at 5 GiB, time to first token rose from 36.39 s to 44.44 s, and process-attributed reads rose from 19.92 GB to 33.30 GB. At `8a6c6347`, four rows also matched every serial logit on real weights, including the first row after prefill. Measure the four-row latency and I/O separately. |
| 3 | Reduce attributed disk reads | The five-prompt-token, twelve-output-token repeated 5 GiB control on `f45062d7` reproduced 1.65 decode tok/s, 9.604 GB of logical expert reads and 6.551 GB of process-attributed disk reads. A 1,536-visit trace reproduced all 3,909 cache misses under byte-weighted LRU. All 4,388 repeated expert selections within the same token’s core passes hit the cache. Across tokens, 3,991 of 6,632 repeated selections hit. Layers 23 and 22 accounted for the most logical reads. Adjacent expert coalescing saved 312 read ranges on the first request and 282 on its repeat, measured against a one-range-per-expert-tensor replay with identical payload bytes. Frequency-based replay increased logical payload reads by 3.7%. The balanced 5 GiB repeat below reproduced a smaller physical I/O reduction. The longer prompt reproduced lower disk reads with higher first-request latency. The first sustained comparison overlapped an iOS CI job. Repeat it in reverse order when the host is idle. Keep mapped storage opt-in. |
| 4 | Measure placement | On the safetensors short repeat, 11 decode steps spent 1.68 s in CPU attention, 3.90 s in expert staging including 2.80 s of demand reads, and 1.11 s in synchronized Metal readback. Real-weight one-row QKV tests favored four CPU workers over the widened Metal path. Measure the end-to-end worker change and record the complete GGUF breakdown. |
| 5 | Verify scaling | The 1 GiB server was rejected at admission. The plan required 3,459,181,568 bytes before inference. The planner correctly applied the 5 GiB, 6 GiB and automatic ceilings on the mini. Recheck latency and swap once the competing iOS CI build has cleared. The larger-device planner checks use simulated budgets. |
| 6 | Qualify each GGUF layout | `6df5ccc4` adds mainline and fork tensor adapters and a CPU reference path. At `4138d0df`, both complete IQ2_XXS artifacts selected token 152717 through the GPU decoder. The fork’s 200,058-logit vector had cosine similarity 0.999983 to the pinned Unsloth runtime and matched its top-ten set. Both layouts passed repeated raw HTTP at `8a6c6347`. The fork’s full-logit comparison at that revision matched all ten leading reference IDs with cosine similarity 0.999974. Verify the later six-format packed Metal kernel and chat through each explicit file path. |

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

### Sustained decode at 5 GiB

The `8a6c6347` server processed 14 prompt and 64 greedy output tokens. Each policy served the same prompt twice in one process, with the mapped process first. All four responses returned identical text about the history of Paris and reached the 64-token limit. The OS page cache was uncontrolled. A local iOS CI worker started during the first request, and Xcode began building during the first private-cache request. These measurements characterize the contention case.

| Request | Dense and expert retention | TTFT (s) | Prefill (tok/s) | Decode (tok/s) | Wall (s) | Logical expert reads (GB) | Attributed disk reads (GB) | Peak process footprint (GB) | Swap-out pages |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| First | Private and LRU | 15.74 | 0.89 | 0.66 | 111.59 | 49.65 | 46.23 | 5.03 | 0 |
| First | Mapped and file cache | 29.07 | 0.48 | 0.60 | 133.94 | 141.63 | 67.18 | 0.46 | 0 |
| Repeat | Private and LRU | 20.18 | 0.69 | 0.55 | 135.79 | 49.51 | 46.39 | 5.03 | 5,312 |
| Repeat | Mapped and file cache | 14.90 | 0.94 | 0.95 | 81.27 | 141.63 | 38.12 | 0.46 | 0 |

Mapped weights with file-cache-only retention increased first-request disk reads by 45.3% and reduced repeat reads by 17.8%. The private repeat overlapped with system swap-out activity. The device runner waits for active local builds to finish before repeating the comparison in reverse order.

### Budget diagnostics with competing host work

The 16 GiB M4 mini reported a Metal recommendation of 12,713,115,648 bytes. For the 128-token context, the shared planner reserved 1 GiB for memory pressure and accounted for 2,385,439,744 bytes of dense weights, state and staging. A configured 5 GiB budget left 1,909,527,552 bytes for packed experts. At 6 GiB the cache limit was 2,983,269,376 bytes. Automatic sizing admitted 9,253,934,080 bytes for experts.

The automatic repeat retained 6,347,464,200 bytes and hit every routed expert. macOS still attributed 3.24 GB of disk reads to the process, with 170,004 system swap-out pages during the request. This run overlapped the local iOS CI build. Record an idle-host comparison before assessing the automatic sizing policy’s impact on swap and throughput. Metal’s recommendation describes GPU resource allocations. [Apple Metal working-set documentation](https://developer.apple.com/documentation/metal/mtldevice/recommendedmaxworkingsetsize)

### Cache experiments awaiting device qualification

`FIFI_INKLING_GGUF_SINGLE_COPY_EXPERTS=1` uses the full already-admitted expert budget for packed Metal buffers when every stored expert projection has a direct packed kernel. Set it alongside `FIFI_INKLING_GGUF_PACKED_MLX=1`. The application retains one copy of each cached packed projection. The phase trace reports the two tier capacities and retained payload bytes. Use the same ceiling with and without the flag when comparing it.

`FIFI_MLX_ADAPT_EXPERT_CACHE=1` applies to the publisher safetensors host cache. The decoder halves its admitted expert retention limit after a macOS warning and clears retained experts after a critical event. It checks the event before staging a visit or when resetting the decoder. The reduced cap lasts until server restart. KV and staging allocations stay within the original plan. Test the no-event path on the mini and capture naturally occurring pressure events before comparing physical I/O. [Apple Dispatch memory-pressure documentation](https://developer.apple.com/documentation/Dispatch/DISPATCH_SOURCE_TYPE_MEMORYPRESSURE)

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
