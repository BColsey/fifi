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

The GGUF inspector recognizes the mainline `inkling2_loop` and repository-root `inkling2` layouts. Both experimental text adapters need the pinned publisher tokenizer. Use the normal `serve --model` command with a directory containing one text GGUF, add `--tokenizer-dir /path/to/publisher-tokenizer`, and set `FIFI_INKLING2_MINI_GGUF_MLX=1` alongside the family opt-in. The fork and mainline use different tensor names and layouts. Qualify each artifact independently.

## Check the serving contract

Use a dedicated server on loopback with the original checkpoint and a context of at least 128 tokens. Start it with a JSON config of this form, substituting the absolute model path.

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
    "context_limit": 128
  }]
}
```

Launch it with the native opt-in and budget from above, using `enginesrv --config /path/to/config.json`. Run the suite from the repository root once the model is ready. Python 3.9 or later is sufficient.

```sh
python3 -B engine/tests/tools/check_inkling_serving.py \
  --base-url http://127.0.0.1:18795 --model inkling \
  --report /path/to/scratch/inkling-serving-evidence.json
```

The report preserves JSON responses and stream events, including reasoning. The test checks raw and chat output caps, streaming parity, the sampled EOS ID and its inclusion in completion usage, explicit multi-turn history, repeatability across requests, and reset after an observed client disconnect. The raw reference check re-encodes the first three matching tokens of the llama.cpp UD-IQ2_XXS and native outputs using the pinned tokenizer. The server exposes sampled token IDs on the chat route.

## Mac qualification TODO

Use the 16 GiB M4 Mac mini for hardware results. The publisher safetensors revision used for these checks is `ed40e7da625f594fc9c4990f5f66be24873c22d7`. The token and context limits below are part of the workload. A fresh server can still read pages from a warm OS file cache.

| Priority | Work | Evidence and remaining checks |
| --- | --- | --- |
| 1 | Verify native text serving | On `f45062d7`, raw HTTP and the WebGUI produced “Paris.” The browser used the model picker and received a completed SSE stream. On `4138d0df`, the follow-up supplied all three messages and displayed “France” with a 144-token output allowance. Both streams ended at EOS. The 68-token control reached its limit before visible answer text. Complete the HTTP contract suite, including disconnect recovery. |
| 2 | Qualify bounded prefill | `c651c820` adds opt-in visit-major prefill. Two-row prefill was bitwise equal to serial for all 200,058 logits at each of six prompt positions and a follow-up. On a repeated 62-token prompt producing eight identical tokens at 5 GiB, time to first token rose from 36.39 s to 44.44 s, and process-attributed reads rose from 19.92 GB to 33.30 GB. Verify the later union-aware implementation at the same ceiling. |
| 3 | Reduce attributed disk reads | The five-prompt-token, twelve-output-token repeated 5 GiB control on `f45062d7` reproduced 1.65 decode tok/s, 9.604 GB of logical expert reads and 6.551 GB of process-attributed disk reads. A 1,536-visit trace reproduced all 3,909 cache misses under byte-weighted LRU. All 4,388 repeated expert selections within the same token’s core passes hit the cache. Across tokens, 3,991 of 6,632 repeated selections hit. Layers 23 and 22 accounted for the most logical reads. Frequency-based replay increased logical payload reads by 3.7%. Repeat the storage comparison below in reverse order and on longer workloads. |
| 4 | Measure placement | Measure CPU projections, output head, routing, attention, demand reads, staging, Metal submission and synchronized readback. Verify numerical parity before changing placement. |
| 5 | Verify scaling | Run automatic sizing, at least two explicit ceilings, and an undersized-server admission test. Check peak physical footprint, swap and the retained cache bound. Treat larger-device plans as simulations. |
| 6 | Qualify each GGUF layout | `6df5ccc4` adds mainline and fork tensor adapters and a CPU reference path. At `4138d0df`, both complete IQ2_XXS artifacts selected token 152717 through the GPU decoder. The fork’s 200,058-logit vector had cosine similarity 0.999983 to the pinned Unsloth runtime and matched its top-ten set. Verify HTTP generation and packed-GPU parity for each layout. Track vision and DSpark independently. |

The comparison controls are opt-in. `FIFI_INKLING_PREFILL_ROWS=1` selects serial prefill. Larger values request up to 16 rows subject to admission. `FIFI_INKLING_EXPERT_TIMING=1` emits per-request cache totals and load, prefill and decode records. The prefill record includes computing logits for the final prompt token. `FIFI_INKLING_ROUTE_TRACE=1` emits each visit’s physical layer, selected and missing experts, cache capacity and read time. Use a separate trace run when measuring throughput.


### Initial 5 GiB storage comparison

These repeated requests used `4138d0df`, the publisher safetensors, a 128-token context, five input tokens and twelve greedy output tokens on the 16 GiB M4 mini. All four runs returned identical text and recorded zero system swap-outs. Each configuration ran in a fresh process with two identical requests. The OS page cache was uncontrolled.

| Dense weights | Packed expert retention | TTFT (s) | Decode (tok/s) | Wall (s) | Expert reads (GB) | Process-attributed disk reads (GB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Private | LRU | 3.41 | 1.52 | 10.66 | 9.60 | 6.70 |
| Mapped | LRU | 4.74 | 1.41 | 12.52 | 9.60 | 7.98 |
| Private | File cache only | 3.85 | 1.48 | 11.28 | 29.43 | 6.17 |
| Mapped | File cache only | 3.60 | 1.56 | 10.65 | 29.43 | 5.78 |

Set `FIFI_INKLING_MMAP_DENSE=1` to map safetensor dense weights and `FIFI_INKLING_FILE_CACHE_ONLY=1` to leave expert retention to macOS. A file-cache hit still contributes to the engine’s logical reads. File-backed pages can also be reclaimed under memory pressure. The fourth row merits further testing across prompt lengths and working-set ceilings.
