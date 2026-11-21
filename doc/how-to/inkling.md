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

The report preserves JSON responses and stream events, including reasoning. The test checks raw and chat output caps, streaming parity, the sampled EOS ID and its exclusion from usage, explicit multi-turn history, repeatability across requests, and reset after an observed client disconnect. The raw reference check re-encodes the first three matching tokens of the llama.cpp UD-IQ2_XXS and native outputs using the pinned tokenizer. The server exposes sampled token IDs on the chat route.

## Mac qualification TODO

Use the 16 GiB M4 Mac mini for hardware results. The publisher safetensors revision used for these checks is `ed40e7da625f594fc9c4990f5f66be24873c22d7`. The token and context limits below are part of the workload. A fresh server can still read pages from a warm OS file cache.

| Priority | Work | Evidence and remaining checks |
| --- | --- | --- |
| 1 | Verify native text serving | On `f45062d7`, raw HTTP and the WebGUI produced “Paris.” The browser used the model picker and received a completed SSE stream. Its follow-up supplied all three history messages and hit the 68-token output cap before visible answer text. Repeat with a larger output allowance and run the full serving suite. |
| 2 | Qualify bounded prefill | `c651c820` adds opt-in visit-major prefill. Two-row prefill was bitwise equal to serial for all 200,058 logits at each of six prompt positions and a follow-up. On a repeated 62-token prompt producing eight identical tokens at 5 GiB, time to first token rose from 36.39 s to 44.44 s, and process-attributed reads rose from 19.92 GB to 33.30 GB. Verify the later union-aware implementation at the same ceiling. |
| 3 | Reduce attributed disk reads | The five-prompt-token, twelve-output-token repeated 5 GiB control on `f45062d7` reproduced 1.65 decode tok/s, 9.604 GB of logical expert reads and 6.551 GB of process-attributed disk reads. A 1,536-visit trace reproduced every cache miss under byte-weighted LRU. Frequency-based replay increased logical payload reads by 3.7%. Compare file-backed dense weights and OS-only retention on matching hardware requests. |
| 4 | Measure placement | Measure CPU projections, output head, routing, attention, demand reads, staging, Metal submission and synchronized readback. Verify numerical parity before changing placement. |
| 5 | Verify scaling | Run automatic sizing, at least two explicit ceilings, and an undersized-server admission test. Check peak physical footprint, swap and the retained cache bound. Treat larger-device plans as simulations. |
| 6 | Qualify each GGUF layout | `6df5ccc4` adds mainline and fork tensor adapters and a CPU reference path. Validate real artifacts through the family loader, sampler, GPU and server before advertising either layout for generation. Track vision and DSpark independently. |

The comparison controls are opt-in. `FIFI_INKLING_PREFILL_ROWS=1` selects serial prefill. Larger values request up to 16 rows subject to admission. `FIFI_INKLING_EXPERT_TIMING=1` emits per-request cache totals and load, prefill and decode records. The prefill record includes computing logits for the final prompt token. `FIFI_INKLING_ROUTE_TRACE=1` emits each visit’s physical layer, selected and missing experts, cache capacity and read time. Use a separate trace run when measuring throughput.
