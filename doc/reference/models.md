# Models

| `family` | Model | Apple Silicon | NVIDIA CUDA | Checkpoint |
| --- | --- | --- | --- | --- |
| `qwen3_6_35b_a3b` | Qwen3.6-35B-A3B | yes | yes | MLX: `mlx-community/Qwen3.6-35B-A3B-4bit`. CUDA: `Qwen3.6-35B-A3B-Q4_K_M.gguf` from `ggml-org/Qwen3.6-35B-A3B-GGUF`. CUDA serves only the GGUF format for this model. |
| `flint` | Flint, 39.8B total, 2.5B active | yes | pilot | MLX: `RecursiveSI/flint-int4`. CUDA: `0xBakeer/TandemLLM-Flint-EXL3-4.0`. |
| `nemotron35` | Nemotron 3.5 Super, 120B total, 12B active | no | yes, greedy only | `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4` at revision `8a115b9d`. |
| `qwen3_8_flash_next` | Qwen3.8-Flash-Next | in the source tree | opt-in | Not validated end to end; A `native-cuda` Linux build serves it on the card unless `flash_next.cuda_decode` is false. |
| `qwen3_8_27b_dense` | Qwen3.8-27B | in the source tree | no | Not validated end to end. |
| `gemma4` | Nemotron Bolt E4B | no | Bolt E4B only | The Bolt E4B Q8_0 GGUF or a K-quant conversion of it; a bare weights file is detected from `general.architecture` = `gemma4`. Other Gemma4 checkpoints are not served. |
| `inkling2_mini` | Inkling 2 Mini | experimental, opt-in | no | [Official NVFP4 and MXFP8 safetensors](../how-to/inkling.md), or a text GGUF (mainline `inkling2_loop` or older `inkling2` layout) |
| `inkling2_loop` | Inkling 2 Mini | CPU pilot | opt-in | Native ModelOpt safetensors and tokenizer files from the [pinned release](https://huggingface.co/thinkingmachines/Inkling-2-Mini/tree/ed40e7da625f594fc9c4990f5f66be24873c22d7). |
| `sensenova` | SenseNova-Loop-9B (text) | CPU only | no | Experimental host path for [SenseNova-Loop-9B GGUF](https://huggingface.co/sensenova/SenseNova-Loop-9B-GGUF). |
| `raon1` | Raon-1-mini-preview, 28.8B total, 4.3B active | no | in the source tree | `raon-1-mini-preview-Q4_K_M.gguf` from `KRAFTON/Raon-1-mini-preview-gguf`. The tokenizer and chat template are read from the GGUF file, so the directory needs no tokenizer files. |
| `fake` | none | yes | yes | Test engine; replies with fixed text. |

"Pilot" means the model serves short contexts and its long-context quality is not measured.

A model path that holds only the GGUF weights file starts from its `general.architecture` metadata: `qwen35moe` selects the Qwen3.6 family and `gemma4` selects Bolt E4B. The other architectures this build knows by name — `sensenova_loop` (pass `--family sensenova`), `qwen4exp`, `nemotron_h_moe`, `dflash` and `clip` — are refused with a note; a `raon1` weights file serves like the tables above describe.

On CUDA, any `.gguf` conversion of Qwen3.6-35B-A3B in `model_dir` serves (the largest when there are several; `Q36_ALLOW_UNPINNED_GGUF=0` serves only the pinned file); the strict open (architecture metadata and full tensor set) is the gate, and the pinned Q4_K_M file stays the qualified checkpoint and wins whenever it is present.

For CUDA GGUF, `qwen36.gguf_sha256` binds startup to an expected full-file digest, including an alternate checkpoint admitted by that switch.
See [content pinning](configuration.md#gguf-content-pinning-cuda).
A content pin does not qualify another quantization or checkpoint for quality.

## SenseNova-Loop

With `ENGINESRV_API_KEY` set, run:

```sh
enginesrv serve --model /path/to/SenseNova-Loop-9B-Q4_K_M.gguf --family sensenova
```

The experimental CPU path serves text with a 4,096-token context. It uses the selected
GGUF's embedded tokenizer and chat template. Leave `--tokenizer-dir` unset; in a JSON
config, set it to the same path as `model_dir`. Set `"x-thinking": "off"` in a chat
request to disable thinking. This path does not use a GPU, vision inputs, MTP draft
decoding, or context and KV overrides.

## Inkling 2 Mini

On a 64-bit build, set `FIFI_INKLING_CPU_REFERENCE=1` and choose family `inkling2_loop` in the `enginesrv` configuration. Use the pinned release directory for `model_dir` and `tokenizer_dir`, or set `model_dir` to a text GGUF in either layout. On a 64-bit Linux build with `native-cuda`, set `FIFI_INKLING_CUDA_REFERENCE=1` instead to run the BF16 and packed expert projections on the first visible CUDA device. Set `CUDA_VISIBLE_DEVICES` to choose the card. Set only one Inkling backend flag. Both pilots support raw completions and plain-text chat, with one active sequence. The default context is 128 tokens including the prompt. Set `FIFI_INKLING_CONTEXT_TOKENS` to opt into a larger context, up to 8,192 tokens on the pinned checkpoint. Routing, attention, KV and convolution state run on the host. On Linux, extended context and prefix storage use FiFi's cgroup-aware host admission. Routing expert page-cache warming receives the RAM budget after the text-state reserve.

`enginesrv check` inspects the safetensors inventory and reports the BF16 KV storage plan for the fixed loop. With a pilot enabled it also checks the pinned tokenizer and the short-context settings. CUDA admission checks available device memory at startup.

For the CUDA pilot, `FIFI_INKLING_CUDA_RESIDENT_EXPERT_MIB` optionally reserves a budget for packed expert banks. The loader places complete banks for the repeated core layers first and includes their bytes in device admission. A budget of zero uses streamed experts. Use `FIFI_INKLING_CUDA_PROFILE=1` to log per-token time by projection and upload phase. Set `FIFI_INKLING_CUDA_DENSE_STREAM=1` to use the experimental vectorized BF16 projections. This path queues all three controller matrices on one stream and copies only the final gain and forget rows back to the host.

The Unsloth Inkling 2 Mini fixed-loop GGUF also works in the text pilot. Point `model_dir` to the GGUF file or a directory with one text GGUF, and use the original Thinking Machines tokenizer files in `tokenizer_dir`. For `enginesrv check`, give the GGUF directory and pass `--tokenizer-dir` separately.

Set `FIFI_INKLING_DSPARK_GGUF` to the separate Inkling 2 Mini DSpark sidecar to opt into speculative decoding. The CUDA backend accelerates the target projections and expert computation with a host drafter by default. Set `FIFI_INKLING_DSPARK_CUDA=1` with the CUDA GGUF target to run draft projections and attention on the GPU. The loader charges draft weights and committed draft KV to the shared state reserve after target placement and checks the remaining runtime headroom. Draft normalization and Markov recurrence run on the host. Set `FIFI_INKLING_DSPARK_CUDA_MARKOV=1` with CUDA DSpark to keep the two Markov projection matrices in VRAM and run those projections on CUDA. Set `FIFI_INKLING_DSPARK_CUDA_HEAD=1` with CUDA DSpark and a resident GGUF vocabulary projection to apply the draft logit correction and select each candidate on the GPU. Set `FIFI_INKLING_CUDA_RESIDENT_ATTENTION=1` to opt into resident target relative attention with a fully resident packed CUDA GGUF target. The host keeps committed KV and convolution histories for prefix restore and verifier rollback. A rejected window repairs the device slots it overwrote that the next token needs. Unset `FIFI_INKLING_DSPARK_GGUF` to return to ordinary decoding.

Set `FIFI_INKLING_CUDA_GROUPED_VERIFY=1` to verify multiple draft rows per visit on the CUDA target. This requires fully resident packed GGUF experts, `FIFI_INKLING_CUDA_BATCH_EXPERTS=1`, `FIFI_INKLING_CUDA_DEVICE_EXPERT_PREFILL=1` and floating-point projection activations. Set `FIFI_INKLING_CUDA_GROUP_ATTENTION=1` and `FIFI_INKLING_CUDA_BATCH_ATTENTION_PROJECTIONS=1` to batch the attention projections, and `FIFI_INKLING_CUDA_BATCH_ROUTER=1` to batch resident router projections. The verifier uses the serial path for one row or when the backend or host budget cannot admit grouped verification.

Both backends map the original packed weights from disk and decode or upload only selected expert payloads. By default, the CUDA backend keeps BF16 dense matrices resident for vectorized CUDA projection and reuses a fixed device buffer for streamed BF16 and NVFP4 experts. In this mode, other supported GGUF quantization types project on the host. Set `FIFI_INKLING_CUDA_RESIDENT_EXPERT_MIB` to cap VRAM for compatible NVFP4 expert banks. The shared expert tier keeps complete banks for repeated core layers first and stops at its VRAM headroom. The remaining experts use the file mapping and bounded staging. Set `FIFI_INKLING_CUDA_STREAM_EXPERTS=1` to stage cold BF16 and NVFP4 expert pairs through FiFi's shared pinned CUDA banks. The GPU decodes the original GGUF NVFP4 blocks directly. Both banks are allocated before hot expert placement.

Set `FIFI_INKLING_CUDA_PACKED_GGUF=1` to use CUDA projections with the original GGUF blocks in all ten Unsloth text variants, from UD-IQ1_S and UD-IQ1_M through UD-Q8_K_XL. FiFi selects the projection by each tensor's GGUF type and uses actual packed sizes to place complete expert layers in VRAM. This option enables pinned staging for the remaining experts. Add the RAM and adaptive-cache settings below to warm mapped expert pages and retain frequently selected experts in VRAM.

Set `FIFI_INKLING_CUDA_BATCH_EXPERTS=1` with packed GGUF projection to batch each resident layer's selected experts. The gate/up projection, clipped activation, down projection, and weighted sum run on one CUDA stream. Each resident layer visit reads back one combined activation row.

During grouped prompt prefill with `FIFI_INKLING_CUDA_BATCH_ROUTER=1` and `FIFI_INKLING_CUDA_DEVICE_EXPERT_PREFILL=1`, set `FIFI_INKLING_CUDA_PREFILL_ROUTER_INPUT_REUSE=1` to gather resident expert inputs directly from the batched router's device input. This skips one hidden-row upload per token when the complete batch fits the existing packed expert weight stage. Router tiles write consecutive rows there; each expert window gathers from its corresponding slice. Routing logits still return to the host for the usual selection, observation and expert scheduling. The option is off by default and does not apply to cold layers or speculative execution. The device expert profile reports `router_input_h2d_bytes_saved` per visit; zero means the ordinary input upload ran.

Set `FIFI_INKLING_CUDA_DIRECT_RESIDENT=1` to keep original GGUF NVFP4 blocks in VRAM and use the direct CUDA kernel for those expert banks.

For VRAM → RAM → SSD expert placement, set `FIFI_INKLING_CUDA_RAM_EXPERT_MIB=auto` or give a positive MiB budget. This enables shared pinned staging and automatic VRAM placement when no explicit VRAM budget is set. After GPU placement, FiFi warms file-backed expert layers within the process's host-memory allowance. Under a smaller allowance, FiFi uses the Qwen expert predictor before attention and asks Linux to read the predicted experts' two GGUF ranges into the page cache. The actual Inkling router determines which experts execute. The budget limits startup warming and advice per visit. Linux can reclaim or retain file pages as memory pressure changes. Set the RAM budget to zero to use ordinary demand paging.

Set `FIFI_INKLING_CUDA_CACHE_EXPERTS=1` to retain frequently routed expert pairs in the remaining VRAM allowance. The shared adaptive policy ranks observed routes per layer. A promotion copies an expert directly from its existing CUDA staging bank. Later hits execute both projections from the cached original bytes. The cache respects the total expert budget and the shared device headroom. With `FIFI_INKLING_CUDA_PROFILE=1`, the server reports hits and avoided host-to-device bytes by layer.

With packed GGUF and adaptive caching enabled, FiFi reserves three quarters of the admitted expert budget for adaptive slots when the full expert weights exceed the budget. The shared device headroom still applies. Set `FIFI_INKLING_CUDA_EXPERT_CACHE_RESERVE_MIB` to choose a fixed quota, or set it to zero to place whole layers first.

Set `"prefix_reuse": true` in the server configuration to reuse saved chat prefixes. Inkling uses the shared scheduler and prefix store with one pinned root and two evictable entries. Each snapshot retains the independent KV and short-convolution state of every visit. KV rows are shared with snapshots while they are in use.

Set `FIFI_INKLING_CUDA_PREFILL_ROWS` from 2 to 16 to group prompt tokens on the GGUF CUDA backend with shared expert streaming. Each attention and convolution state advances in token order. The CUDA staging banks upload each missing expert once per layer visit and prompt chunk. `FIFI_INKLING_CUDA_PROFILE=1` records expert uploads in the shared request log. A value of 1 uses single-token prefill and is the default.

Set `FIFI_INKLING_DSPARK_GROUPED_PREFILL=1` to capture target features during grouped prompt prefill on the same compatible CUDA backend as grouped verification. Set `FIFI_INKLING_CUDA_PREFILL_ROWS` above one to request a group width. The host memory planner admits the width, and the startup prefill report shows the admitted value. The prefill and verification switches can be enabled independently. Token-major feature capture is the default.

Set `FIFI_INKLING_CUDA_CHAIN_ATTENTION_OUTPUT=1` with `FIFI_INKLING_CUDA_RESIDENT_ATTENTION=1` to keep the attention result on CUDA through its resident output projection. The engine reads back the projected activation before the existing causal convolution. A projection without eligible resident weights uses the existing path.

Set `FIFI_INKLING_CUDA_PREFILL_Q8_ROW_REUSE=1` to opt into paired F32 activation rows for resident Q8_0 attention projections on SM75. It applies only to existing multirow prefill: `(M,N,K)=(7–12,1024,2304)`, `(7–12,3072,2304)`, or `(63–64,2304,3072)`. It does not enable batching or affect decode (including grouped speculative verification), routed experts, or WMMA. `FIFI_INKLING_CUDA_PREFILL_Q8_ROW_REUSE_REPORT=1` reports paired and ordinary Q8_0 enqueues for dispatch checks. See the [component notes](../../engine/tools/inkling-q8-row-reuse/README.md) for prerequisites and measurement limits.

Set `FIFI_INKLING_CUDA_PREFILL_QUERY_TILES=1` to opt in to query-parallel relative attention on SM75 for ordinary grouped GGUF prefill. This requires resident attention and output projection, `FIFI_INKLING_CUDA_BATCH_RELATIVE_ATTENTION=1`, and admission to the existing batched attention path. Eligible groups of 2–64 queries read one immutable device-ring snapshot and then commit their K/V in stream order. Host K/V remains authoritative; prefix restore and reset invalidate the device cursor. Singletons, unsupported hardware and ineligible batches use the existing path. The flag is off by default and does not apply to decode or speculative verification.

Set `FIFI_INKLING_CUDA_PREFILL_RESIDENT_QUERY=1` to keep Q, relative coefficients, sinks and gates on CUDA for eligible ordinary grouped GGUF prefill. This is off by default and requires the existing batched relative-attention path and resident output projection; it does not enable batching, controller chaining or the resident cache by itself. The host still prepares and retains K/V. With query tiles enabled and admitted, query preparation and snapshot attention share status so an invalid row prevents the whole tile's K/V commit. Otherwise resident query preparation feeds the existing serial attention launches. Ineligible paths fall back before submission; an enqueued batch error is not retried on a potentially modified ring.

Set `FIFI_INKLING_CUDA_BATCH_CONTROLLERS=1` with `FIFI_INKLING_CUDA_GROUP_PROJECTIONS=1` to group resident Q, K, and V cross-pass controller projections in one transfer. Grouped prefill also batches those controllers across prompt rows when `FIFI_INKLING_CUDA_BATCH_ATTENTION_PROJECTIONS=1` is set. Controllers with missing resident matrices use their existing fallback.

Set `FIFI_INKLING_CUDA_CHAIN_ATTENTION_INPUT=1` with `FIFI_INKLING_CUDA_BATCH_CONTROLLERS=1`, `FIFI_INKLING_CUDA_GROUP_PROJECTIONS=1`, and `FIFI_INKLING_CUDA_GROUP_ATTENTION=1` to keep cross-pass controller outputs on CUDA through residual gating, weighted RMSNorm, and the attention input projections. This applies to ordinary single-row GGUF execution when all required matrices are resident. Grouped prefill and speculative verification retain their existing paths. The chain is off by default.

Set `FIFI_INKLING_CUDA_CONTROLLER_CPU_PARITY=1` to experiment with host-style arithmetic only in the fused Q/K/V controller normalization for eligible grouped GGUF CUDA prefill (`FIFI_INKLING_CUDA_GROUPED_CHAIN_ATTENTION=1`). The parity flag does not enable the chain or change its eligibility or fallback. Single-row fused attention (including resident local decode) and routed MLP controllers always use the generic CUDA arithmetic. The flag is read once when the grouped fusion is first admitted; unset it (the default) to use generic arithmetic there too.

The parity kernel replaces device `tanhf` with device FP64 `tanh` rounded to FP32, and uses host-ordered FP32 sum of squares, square root, reciprocal, and gamma multiplication. On one exact-model first tile with identical inputs, both rounded tanh taps and the pre-norm activations matched the host at all 20,736 elements each. A standalone replay through the fused kernel also matched all 20,736 normalized outputs on that tile. This does not establish tanh equality on other inputs or hosts, or whole-model parity.

Set `FIFI_INKLING_CUDA_CHAIN_LOCAL_DECODE=1` with `FIFI_INKLING_CUDA_GROUP_ATTENTION=1` and `FIFI_INKLING_CUDA_RESIDENT_ATTENTION=1` to run local decode attention projections, causal K/V convolution, clipping and normalization, and relative attention before one readback. Enable the controller input chain above to include cross-pass local visits. The readback still commits raw convolution history and processed KV rows to the host for snapshots and rollback. Global attention, grouped prefill, draft-enabled sessions, missing resident weights, and visits whose device cursor has not been restored use the existing path. The chain is off by default.

Set `FIFI_INKLING_DSPARK_ADAPTIVE_DRAFT=1` to reduce the DSpark draft width after a rejection and restore it after fully accepted windows. Two consecutive windows accepting fewer than half their proposals switch that generation to one-token target decode. Each new request starts at the configured width, including a request that restores a saved prefix. The request log reports adaptive stand-down steps when they occur. The fixed DSpark width is the default.

## Tokenizer files

A `raon1` engine reads its tokenizer and chat template from the GGUF file instead.
Every other engine reads `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` and `generation_config.json` from its `tokenizer_dir`.
For the three validated families, startup compares the files with the SHA-256 hashes of the tested release and refuses others.
For Qwen3.6 CUDA, use the tokenizer files from `mlx-community/Qwen3.6-35B-A3B-4bit` with the GGUF file. The other validated checkpoints listed above include matching files.

## Weight formats

| Format | Used by |
| --- | --- |
| MLX affine-quantized safetensors; the quantization is read from `config.json` | Qwen3.6, Qwen3.8-27B and Flint on Apple Silicon |
| GGUF with F16, Q4_K, Q5_0, Q6_K and Q8_0 tensors | Qwen3.6 on CUDA |
| GGUF Q8_0, and K-quant conversions of it | Bolt E4B on CUDA |
| GGUF, matrices dequantized on demand | SenseNova-Loop on CPU (the published Q4_K_M file) |
| GGUF, mainline `inkling2_loop` or older `inkling2` layout | Inkling 2 Mini on Apple Silicon, and on CPU or CUDA with family `inkling2_loop` |
| NVFP4 safetensors | Nemotron 3.5 Super |
| Mixed NVFP4 and MXFP8 safetensors | Inkling 2 Mini on Apple Silicon |

## Context limits

| Model and backend | Default | Largest |
| --- | ---: | ---: |
| Qwen3.6, Apple Silicon | 262,144 | 262,144 tested on a 36 GB Mac |
| Qwen3.6, CUDA | 32,768 | 262,144 tested on a 24 GB RTX 3090 |
| Flint, Apple Silicon | 32,768 | 32,768 |
| Flint, CUDA | 131,072 | 262,144 |
| Nemotron 3.5 Super | set in config | |
| SenseNova-Loop, CPU | the GGUF's declared context, capped at 4,096 | 4,096 |
| Inkling 2 Mini, Apple Silicon | 1,024 | Configurable up to the model limit of 1,048,576, subject to admission |

## Recent updates

- **Opt-in on-demand KV for Qwen3.6 GGUF on CUDA.** With `Q36_TIERED_KV_ON_DEMAND=1` (default off), a single-card sessions start funds a small initial cell count instead of the full context ceiling and grows the cache as positions are fed. The flag changes nothing else: starts that do not plan sessions ignore it, and a named tiered member still refuses `max_seqs`.
- **Bolt sliding-window ring layout.** A sliding layer's cache is a ring: position `p` lives in cell `p % ring`, so a session's cache grows with the full-attention layers only. Disk prefix snapshots moved to version 2 (`bolt-kv8-snapshot-v2`): a snapshot keeps every position of the full-attention layers and only the last window of each sliding layer, in position order. A version 1 entry holds every position of every layer and is never read as a version 2 one.
