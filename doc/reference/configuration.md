# Configuration file

`enginesrv --config <file>` reads one JSON file.
Startup refuses unknown keys and values outside the ranges below.
When a model does not fit in memory, startup refuses the load rather than lowering a setting to make it fit.

Example files are in [`engine/config`](../../engine/config/README.md).

## Top level

| Key | Required | Default | Meaning |
| --- | --- | --- | --- |
| `listen` | yes | | Address and port, for example `127.0.0.1:18791`. The address must be `127.0.0.1` or `::1`. Port 8788 is refused. |
| `api_key_env` | yes | | Name of the environment variable that holds the API key. Startup fails if the variable is unset or empty. |
| `prefix_reuse` | no | unset | Keeps finished conversation prefixes so that a follow-up turn processes only its new tokens. Unset, the start decides per family: on for Flash-Next, Qwen3.6 and Bolt on CUDA; on for Qwen3.6 on Apple Silicon when host memory holds the model, the session reserve and one saved prefix at the start's context (the start logs the numbers); off for every other family. Saving a prefix splits the prefill in two pieces, which can change greedy text through the order of the floating-point sums. `true` or `false` overrides for all engines. See [How conversation reuse works](../explanation/prefix-reuse.md). |
| `web_dir` | no | | Directory with the built web frontend (`npm run build` in `web/` writes `web/dist`). When set, `GET /` serves its `index.html` and `GET /assets/...` its hashed assets; startup refuses a directory without `index.html`. The shell is public like `/health`; every API route keeps its key. |
| `engines` | yes | | The engines to serve, one object each. |

## Engine entry

| Key | Required | Meaning |
| --- | --- | --- |
| `id` | yes | Engine name. Clients can select the engine with it (`x-engine`). |
| `family` | yes | The model family; see [Models](models.md). |
| `model_dir` | yes | Directory of the checkpoint. It must be on local storage; see [Local storage](#local-storage). |
| `tokenizer_dir` | yes | Directory with `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` and `generation_config.json`. For Qwen3.6, Flint and Nemotron, startup checks each file against the hashes of the tested release. |
| `model` | no | The name clients send in the `model` field. Defaults to the family name. |
| `qwen36` | no | [Qwen3.6 options](#qwen36). |
| `nemotron` | `nemotron35` only, required | [Nemotron options](#nemotron). |
| `fake` | no | [Test engine options](#fake). |
| `context_limit` | no | `fake` only: context limit in tokens. Default 4,096. |
| `flash_next` | no | Options of `qwen3_8_flash_next`, including [MLX prefill admission](#flash-next-mlx-prefill). The family is not validated end to end. |
| `think_replay` | no | `false` drops the reasoning that clients resend with earlier assistant turns from the prompt, for this engine only. `true` is refused; leave the key out for the default. |
| `notice_toolwrap` | no | `false` renders system messages that follow an assistant turn as plain system text, for this engine only. `true` is refused. See `ENGINESRV_NOTICE_TOOLWRAP`. |

### Local storage

Model weights must be on local storage; network storage is not a tier or a fallback. The server refuses to start an engine whose `model_dir` is on a network filesystem (NFS, SMB, AFP, WebDAV, sshfs, Ceph and similar), or on an automount that has mounted nothing. The loaders also refuse a shard file that is a symlink into such a mount, because a symlink counts as where it points. The error reads `model weights must be on local storage, but <path> is on a <type> filesystem`. If the filesystem cannot be determined, the error reads `cannot confirm that model weights at <path> is on local storage` and the engine does not start. The tokenizer directory holds only small text files and is not checked.

### CUDA FP16 prefill attention

FP16 prefill attention retains the existing PTX-based launch path by default. Set [`FIFI_CUDA_ATTN_PREFILL_ADAPT=1`](environment.md#cuda) before engine startup to opt into selection for the current CUDA device and model geometry. The adaptive path uses the loaded kernel's PTX version, the device's per-block shared-memory limit and the device and kernel thread limits. It selects the tensor-core arm when it fits and its loaded code supports it, otherwise the compact SIMT arm when that fits. If neither fits, or a device query or shared-memory opt-in fails, the launch returns an error. Successful adaptive selections are cached per device and geometry.

This selection applies to the FP16 attention part kernel, not every model kernel or the q8 attention path. It does not establish correctness or performance for every GPU and model. The selection was tested on an RTX 2080 Ti and an RTX 3080; RTX 40 and 50 series boards were not tested. The build must also contain compatible device code; see [`INFERENCE_CUDA_ARCHS`](environment.md#build).

## Flash-Next MLX prefill

An optional positive integer, `flash_next.prefill_budget_bytes`, gives the engine a total byte budget for batched prompt processing. Omit the key to serve one token at a time. With a budget, startup selects the largest fitting chunk up to `flash_next.prefill_chunk` (default 16, range 1–2,048). A worker piece below four rows or insufficient room for a four-row chunk selects the one-token path. The budget applies to chunk admission after model load. This setting is specific to MLX.

The startup line gives the selected path and the bytes charged. The planner reserves resident weights, the decode pool, the QSA cache at the full 32,768-position limit, fixed recurrent state, the configured MLX allocator cache and a 1 GiB pressure margin. With prefix reuse enabled, it also reserves full-sized snapshots for the prefix store's slots by default. Local shard reads have a zero S3 cache charge. The remaining bytes determine which chunk workspace fits.

`FIFI_FLASH_NEXT_MLX_PREFILL_WAVE_CAP` bounds how many routed experts each MLX prefill wave can hold. It accepts an integer from 1 to 512 at startup and defaults to 512. The same cap determines both the workspace charge and the staging pool size. A lower cap can admit a wider chunk within the device budget. Layers whose selected experts exceed the cap run multiple waves and preserve the router's original pair order. Startup logs the wave cap and the selected chunk's workspace charge in bytes.

With prefix reuse enabled, set `FIFI_FLASH_NEXT_MLX_PREFIX_BUDGET_BYTES` to a positive byte count to bound the retained snapshot store. For example, `268435456` sets a 256 MiB cap. Startup charges the smaller of this cap and the default snapshot reservation. A snapshot that exceeds the cap is skipped and previously retained snapshots remain available. The planner continues to charge live state, restore overlap and the allocator cache separately.

`FIFI_FLASH_NEXT_MLX_DECODE_CACHE_SLOTS` sets the number of retained expert records per layer during sampled decode. It defaults to zero and accepts up to four slots, bounded by the model's experts selected per token. A positive setting requires automatic MLX sizing or an explicit `FIFI_MLX_WORKING_SET_MIB` ceiling. Startup checks the page-rounded cache mapping and its metadata against the device budget. The mapping is released at the decode boundary before the next prefill. Only demand reads populate it, after an expert is selected twice in a row. Cached slices are copied into the existing rank-ordered staging set.

## `qwen36`

Options for `qwen3_6_35b_a3b`. "MLX" means Apple Silicon.

| Key | Backend | Default | Meaning |
| --- | --- | --- | --- |
| `context_limit` | both | MLX 262,144; CUDA GGUF: the largest the free card carries, up to 262,144 | Context limit in tokens, from 2. A CUDA GGUF start on one device with no `tiered` and no `kv_offload_host_bytes` reads the free card before the weights are uploaded and takes the largest context (a multiple of 1,024) whose cache, DeltaNet state, staged int8 rows and admission margins fit beside the weights; the start line names the choice. A value you set is served as given. When the card cannot carry even the smallest context the start keeps 32,768 and takes the tiered rung or the load's own refusal. |
| `kv_dtype` | both | MLX `q8`; CUDA GGUF: chosen from the free card | KV cache format: `q8` (8-bit), `f32`, or `bf16` (MLX only). On CUDA, `f32` stores the cache in FP16. On the CUDA GGUF start described under `context_limit`, an absent `kv_dtype` is chosen with the context: FP16 when it reaches 131,072 positions (or the context you set), otherwise `q8`, which holds almost twice the positions in the same bytes. `q8` changes generated text against FP16; name `kv_dtype` to fix the form. |
| `activation_dtype` | MLX | `bf16` | Decode activations: `bf16` or `f32`. CUDA accepts the key and ignores it. |
| `cache_limit_bytes` | MLX | none | MLX allocator cache limit. Falls back to `Q36_CACHE_LIMIT`; on MLX one of the two is required. |
| `session_cap_bytes` | MLX | 4 GiB | Memory for the live session and saved prefixes. |
| `prefill_budget_bytes` | MLX | off | Memory budget for processing the prompt in chunks. Without it, the prompt is processed one token per step. |
| `prefill_session_reserve_bytes` | MLX | derived | Memory that prompt processing leaves free. With `prefill_budget_bytes` and no reserve member, it is derived: the attention cache at `context_limit`, plus the DeltaNet state, plus `cache_limit_bytes`, plus a 1 GiB pressure margin (the shipped Mac profile's composition). An explicit value always wins. Refused without `prefill_budget_bytes`. |
| `prefill_chunk` | MLX | 256 | Largest prompt chunk, 1 to 2,048 rows. The engine picks the largest chunk that fits the budget. |
| `compile` | MLX | `false` | Compile the decode step with `mlx_compile`: 9.6 to 13% faster decoding. Logits differ from the uncompiled path by at most 5e-5. |
| `compile_program_limit` | MLX | 8 | Budget of distinct compiled program keys per adapter, from 1. Increasing it is opt-in and does not enable `compile`. CUDA refuses an explicit value. |
| `max_seqs` | MLX | 4 on a host-derived MoE start with compiled decode and a funded prefill budget, else 1 | Sequences decoded at once, 1 to 8. Above 1 it needs `compile: true`, a `prefill_chunk` of at least 4, and a `prefill_budget_bytes` larger than both `prefill_session_reserve_bytes` and `session_cap_bytes`. A derived value that the memory plan refuses starts one session instead, and the start says so; a value you set is never lowered. Clients decode together in one window. A client's greedy text in a window can differ from its text alone: the window is held to the batch quality gate (mean KL, worst-row KL and top-1 agreement against the solo run), not to bit equality. |
| `prefix_store_entries` | MLX | 6 | Saved prefixes kept with `prefix_reuse` on, 1 to 4,096. |
| `weight_format` | CUDA | `gguf` | The format of the checkpoint in `model_dir`. `gguf` serves the pinned Q4_K_M file when it is present, otherwise the directory's largest `.gguf` of the family, validated at open; `Q36_ALLOW_UNPINNED_GGUF=0` serves only the pinned file. Any other value is refused, and `exl3` is refused by name. |
| `gguf_sha256` | CUDA GGUF | off | Expected SHA-256 of the selected GGUF file, exactly 64 hex digits. Startup always verifies this pin before opening the model. MLX refuses the key. See [content pinning](#gguf-content-pinning-cuda). |
| `capture` | CUDA | `false` | Run each decode step as a CUDA graph. |
| `cuda_devices` | CUDA | `[0]` | The CUDA devices this engine spans, as device numbers in strictly ascending order, at most four. The MLX arm refuses the key. A list other than `[0]` is probed at startup (the devices must exist, share one compute capability, and directly access each other's memory) and serves the whole model across the cards, GGUF checkpoints only, with `tiered`, `kv_offload_host_bytes`, `capture` and speculation off. |
| `kv_offload_host_bytes` | CUDA | off | Keep the KV cache in host memory, up to this many bytes. See [below](#host-memory-kv-cache-cuda). |
| `kv_cache` | CUDA | on, in the user cache directory | An object, at the engine, that keeps saved prefixes on local disk, so a long conversation survives a restart and an eviction from pinned memory. Members: `dir` (an absolute path), `quota_bytes` (from 1 GiB) and `enabled` (`false` turns the cache off). See [below](#prefix-cache-on-disk-cuda). |
| `tiered` | CUDA | off | Keep only some experts on the GPU. See [below](#tiered-experts-cuda). |
| `draft_mode` | MLX | `disabled` | Where speculative drafts come from: `disabled`, `ngram` (the tokens that followed the latest earlier occurrence of the context's tail) or `mtp` (the model's own MTP head, which needs `mtp_model_dir`; not usable yet). Any other value is refused. |
| `num_draft_tokens` | MLX | 2 | The most drafts one verify window carries, 1 to 16. |
| `dynamic_draft_tokens` | MLX | `false` | Shorten each window from the measured chance that its drafts are accepted. Reads the MTP head's confidence, so `ngram` ignores it. |
| `draft_confidence` | MLX | 0.4 | With `dynamic_draft_tokens`: a draft is dropped when the product of the acceptance chances of the drafts so far falls below this value, above 0 and below 1. |
| `draft_max_rows` | MLX | 8 | The most rows one window feeds, the pending token included, 2 to 64. It caps `num_draft_tokens` and `hybrid_draft_tokens`. An enabled `draft_mode` refuses more than 8. |
| `hybrid_draft_tokens` | MLX | 0 (off) | When the context repeats an earlier passage for at least this many tokens, draft that many tokens of the passage in one window. 0, or from `ngram_match_min` to 256. |
| `ngram_match_min` | MLX | 2 | The shortest repeated tail that an n-gram draft needs, 1 to 64. |
| `mtp_model_dir` | MLX | none | The pack directory that holds the `mtp.*` tensors. Required by `draft_mode` `mtp`, refused otherwise. |
| `draft_rollback` | MLX | `host` | How a verify window is undone after a partial accept: the committed state is copied to the host, restored, and the kept rows are fed again. Any other value is refused. |
| `draft_verify` | MLX | `eager` | How a verify window runs its rows. `eager` feeds the rows through the one-row step, one after the other. Any other value is refused. |

CUDA refuses `prefill_budget_bytes` and `prefill_session_reserve_bytes`.
It ignores `cache_limit_bytes`, `session_cap_bytes`, `prefill_chunk`, `compile` and `prefix_store_entries`, and logs one line for each.
MLX ignores `capture` with a log line, and refuses `tiered`.
The `draft_*`, `num_draft_tokens`, `hybrid_draft_tokens`, `ngram_match_min` and `mtp_model_dir` keys are checked at start. A Qwen3.6 start on the CUDA arm that sets any of them is refused with `engine config: Qwen3.6 CUDA speculative verify is removed until the step's many-rows path brings it back; unset the draft options`. On MLX, a `draft_mode` other than `disabled` needs GGUF weights, and `mtp` is refused because its draft step does not exist.
`prefix_store_entries` is ignored with a log line when `prefix_reuse` is off.

Before saving a new host-backed pageable prefix (the opt-in pageable path; the default pinned-prefix path is unchanged), the CUDA engine checks live host and process-cgroup memory. The check covers the committed KV and recurrent state on every participating card, the largest temporary copy buffer, and 1 GiB of headroom. If that space is unavailable, the engine skips the optional save with one logged line (what was needed, what was available, that the request is unaffected) and keeps the already-cached prefixes available for later requests; if no budget figure can be read at all, the save proceeds as before with one logged line. Reusing an existing checkpoint requires no new allocation and passes without a probe.

### Prefix cache on disk (CUDA)

`kv_cache` keeps the saved prefixes of a CUDA engine in a directory, so the next request of a long conversation reads its state from disk in seconds instead of processing the prompt again, also after a restart. It is one key at the engine, the same for every family that keeps saved prefixes in host memory: the Qwen3.6 GGUF engine and the Bolt engine take it; a family without such a cache refuses the key. (The Qwen3.6 and Bolt options objects also accept it as `qwen36.kv_cache` and `gemma4.kv_cache`; naming it twice is refused.)
It is on by default for a Qwen3.6 GGUF engine on one CUDA device and for the Bolt engine. It needs the family's host prefix store (`prefix_reuse` and, for Qwen3.6, the pinned prefix pool); with that off, or with more than one CUDA device, or when the host has too little free space, the cache stays off and says why in its start line. The Qwen3.6 cache serves GGUF loads only.

While the cache is on it writes the conversations the engine has served (their exact token lists and the model state after them) to disk in `dir`, up to `quota_bytes` (at most 64 GiB by default). To keep nothing on disk, set `kv_cache.enabled` to `false` or `FIFI_KV_CACHE=0`; `--purge-kv-cache` removes what is there. The start line says all of this.

```json
"kv_cache": { "dir": "/var/cache/fifi/kv", "quota_bytes": 21474836480 }
```

- `dir`: absolute. Created with mode 0700 when missing; an existing directory must be owner-only. One process owns a directory at a time. Without `dir` the cache uses `$XDG_CACHE_HOME/fifi/kv/<family>`, else `$HOME/.cache/fifi/kv/<family>` (`qwen36`, `gemma4`), so two families never share a directory by default. Two engines of one family need two directories.
- `quota_bytes`: the most the directory may hold, at least 1 GiB. Without it the quota is a quarter of the free space of the directory's filesystem at start plus the bytes the directory already holds (so a disk that has filled up since does not strand them), at most 64 GiB; below 4 GiB the cache stays off and the start line says how many bytes the directory still holds and that `--purge-kv-cache` empties them. A quota below what the directory holds removes the least recently used entries at start. The least recently used entry goes first.
- `enabled`: `false` keeps the cache off whatever else is set, including `FIFI_KV_CACHE_DIR` (only `FIFI_KV_CACHE=1` turns it back on); `true` turns it on at the default directory.

The environment overrides the config: `FIFI_KV_CACHE_DIR`, `FIFI_KV_CACHE_QUOTA_BYTES`, `FIFI_KV_CACHE_HOST_BYTES` (the bytes of saved prefixes the host store may hold before the oldest go to disk, see below), and `FIFI_KV_CACHE=0` or `1`, with the exception of `enabled: false` above. The same names with a `Q36_` prefix instead of `FIFI_` still work. A malformed value stops the start. The directory must support direct I/O (a directory on tmpfs, some overlay and encrypted file systems does not); the start writes a scratch file of 8 MiB there with direct I/O, syncs it, reads a sector back and deletes it, which both checks the directory and times the disk, and it keeps the cache off with the reason when the directory does not support it.
`enginesrv --config <file> --purge-kv-cache` empties the cache directory of every engine in the config whose family keeps one (only a directory that holds a cache) and then starts.

An entry holds the exact tokens and the state of one saved prefix: for Qwen3.6 the DeltaNet state and the KV rows, for Bolt the KV8 cells of every physical layer.

The disk is used only when host memory is short. When the host's available memory falls under the headroom the start plan keeps plus one more entry, and the store has reached its watermark, the least recently used entries are written in the background. A host with memory to spare writes nothing while serving. A background write whose slot a save needs first is dropped and counted.

A graceful stop (SIGINT or SIGTERM) writes the tips that exist only in memory, newest first. The wait is bounded: three times the time those bytes need at the measured write rate, plus 10 s, between 15 s and 300 s. The stop line says when the bound is longer than systemd's default 90 s or Docker's default 10 s stop timeout. A second signal ends the wait at once.

A write leaves 512 MiB of the filesystem free and is skipped otherwise. Entries below 1,024 positions are not written.

An entry is read back on a host-store miss, with a CRC32C and SHA-256 check of every byte. Each entry carries the identity of the load that wrote it: a digest of the GGUF header, a sample of every tensor, the file length and modification time, the `gguf_sha256` pin when there is one, the tokenizer and chat-template digests, the KV format and a version of the numeric recipe. An entry of another load, or a truncated or corrupted one, is a logged miss, and the prompt is processed as without the cache.
The disk is also used when the saved prefixes in host memory outgrow their byte budget: past it the least recently used are written to disk in the background, the pinned roots last (a written entry stays in memory until the store evicts it). The budget is the planner's room for saved prefixes under a device budget (Qwen3.6 on Apple silicon), and otherwise the machine's memory less the OS floor and less what the engine already holds resident, read at each save, so every family has one without a setting; `FIFI_KV_CACHE_HOST_BYTES` replaces it with a byte count, and a malformed value stops the start. The start line states the budget.
The request log reports `prefix_source` (`device`, `pinned` or `disk`), for a disk restore `prefix_disk_bytes` and `prefix_disk_ms`, and the totals written since the start: `prefix_disk_written_entries`, `prefix_disk_written_bytes` and `prefix_disk_dropped`. The start line says where the watermark is, and each write says why (`watermark`, `byte budget` or `stop`). After three failed writes in a row (a full or read-only disk) the cache stops writing for ten minutes, with one log line.

### GGUF content pinning (CUDA)

Set `qwen36.gguf_sha256` to a trusted full-file SHA-256 to bind startup to specific checkpoint bytes.
The value accepts uppercase or lowercase hex digits.
A mismatch refuses startup before model parsing or weight upload.
This explicit pin takes precedence over `Q36_VERIFY_GGUF_SHA256`, so startup hashes the file once.
Without the key, the existing filename, size and optional built-in hash checks remain unchanged.

The key also works for an alternate checkpoint filename.
It does not enable unpinned admission or change which file startup selects.
The pinned filename still wins when present, and its size check still applies.
A matching digest establishes content identity, not model quality or source authenticity.
Keep the checkpoint files unchanged while the engine serves them.

### Host-memory KV cache (CUDA)

`kv_offload_host_bytes` keeps the KV cache in host memory for contexts that do not fit on the GPU.
It needs `"kv_dtype": "q8"`, `"capture": false`, `"prefix_reuse": false` and `Q36_CUDA_CHUNK_PREFILL=0`; startup refuses the combination without `prefix_reuse` false, so on the CUDA build an omitted `prefix_reuse` key (which defaults on for Qwen3.6) also refuses `kv_offload_host_bytes` until the key is set to false explicitly.
It reloads the cache from host memory for every token, so it is slow: an 8K-token replay decoded at 34.95 tokens/s, against 147.14 tokens/s with the cache on the GPU.
`Q36_CUDA_PAGED_DEVICE_FIRST=1` keeps as much of the cache on the GPU as fits after a 1 GiB reserve.

### Tiered experts (CUDA)

`tiered` keeps a fixed number of expert slots per layer on the GPU and runs the other experts on the CPU from the memory-mapped model file.
Do not modify or replace the model file while the server is running.

| Key | Default | Meaning |
| --- | --- | --- |
| `slots` | required | An integer from 8 to 256 requests an exact count. `"auto"` selects the largest whole number that fits every layer after weights, KV cache and step allocations, leaving 1 GiB of free VRAM. Startup reports the selected count and refuses a load that cannot hold one token’s routed experts. |
| `threads` | 6 | CPU worker threads, 1 to 8. |
| `timeout_ms` | 10,000 | Longest wait for one layer's CPU experts, 1 to 600,000. |
| `preload` | `true` | Fills each layer's slots with its lowest-numbered experts before the first token, so the start reports ready with a full bank. `false` starts the slots empty; they then fill as experts are used. |
| `swaps_milli` | 250 | Bank swaps per token, in thousandths, 0 to 8,000. The default 250 buys one promotion per four tokens model-wide — sized for the blocking fill path. |

Tiered mode refuses `capture: true` and `kv_dtype: q8`.
It processes prompts one token per step and, by default, keeps no saved prefixes in host memory.

## `flash_next`

The `flash_next` object of one engine entry for `qwen3_8_flash_next`. Every key is optional. The
keys below are the CUDA prefill keys; they need a `native-cuda` Linux build. That build serves on the
card unless the entry sets `cuda_decode: false` (the MLX path, where the CUDA keys are refused).

| Key | Default | Meaning |
| --- | --- | --- |
| `cuda_decode` | `true` in a `native-cuda` Linux build unless `cache_limit_bytes` or `prefill_budget_bytes` is set, `false` elsewhere | Serves the model on a CUDA card. The CUDA decoder is experimental. |
| `prefill_chunk` | 16 | Rows per chunked-prefill piece, 1 to 2,048. A piece wider than `prefill_rows` is streamed in `prefill_rows`-row device chunks (four per 1,024-row piece at the default `prefill_rows` of 256). Wide pieces need the prefill ring (`INFERENCE_FLASH_NEXT_RING_BUDGET`) to be useful. Single measured runs on the RTX 3090 EXL3 pack, a 28,916-token prompt, ring on, `prefill_rows` 256 (the default): prefill took 229 s at `prefill_chunk` 256, 221 s at 1,024 and 219 s at 2,048 - the chunk size barely moves it while the device step stays 256 rows. |
| `prefill_rows` | 256 | Rows of one device prefill step, 1 to 2,048 (the shared supported width; the unset default is 256, and `1` selects the per-token prefill). Measured single runs, RTX 3090 EXL3 pack, the same 28,916-token prompt, ring on (budget 1.3e9), `prefill_chunk` 2,048, no measurement override set: 256 rows prefill 234.9 s / 3,390 GB streamed / 804 misses / 25.08 decode tok/s; 1,024 rows 184.1 s / 943 GB / 861 misses / 23.52 decode tok/s; 2,048 rows 180.1 s / 540 GB / 921 misses / 23.16 decode tok/s (all unwrapped). An earlier 1,024-row run under the nsys profiler measured 162 s prefill and 25.98 decode tok/s - profiler-wrapped walls are comparable only to other wrapped runs; the unwrapped figures above are the served numbers. The ring held 658 slots (1.15 GiB) at 256 rows and 690 (1.21 GiB) at the wider windows. The wider window's arena is card memory the expert bank gives up: resident experts per layer fell 183 (256 rows) to 165 (1,024) to 141 (2,048), and decode pays for it - 25.08 tokens per second at 256 rows against 23.52 at 1,024 and 23.16 at 2,048. A window the card cannot carry next to the trunk refuses at load with the losing budget's byte counts. |
| `context` | from the memory plan | K/V positions, a multiple of 4 from 4 to 262,144. Unset, the start picks the longest context, up to the model maximum, at which the expert bank keeps nine tenths of the slots it has at 4,096, and prints the choice and the slots on one line. A value set here is used as given: the load refuses a context the card cannot hold next to the trunk. |
| `kv_format` | `f16` | K/V storage on the card. `int8` has no attention kernel yet and is refused. |

## `nemotron`

Required for `nemotron35`. `cache_entries` may be omitted; the other three keys are required.

| Key | Values | Meaning |
| --- | --- | --- |
| `backend` | `"cuda"`, `"cpu_reference"` | `cuda` needs a Linux build with `--features native-cuda` and accepts greedy decoding only. `cpu_reference` runs on the CPU and is slow. |
| `context_limit` | 1 or more | Context limit in tokens. |
| `cache_entries` | 1 or more, or omitted | Expert weight tensors kept decoded in host memory. Admission charges each entry as the largest entry's payload bytes plus the loader's per-entry index, 96 bytes per entry on 64-bit targets. Omitting the key fits the count from 850 permille of the measured host budget (the cgroup-aware figure on Linux) minus the fixed serving charges; a remainder below one entry refuses the load, and one log line states the chosen count. An explicit number is honoured verbatim: a card too small for it refuses the load - omit the key, because only the fit trims the 12 GiB device request down to what the card holds. |
| `prefill_chunk` | 1 to `context_limit` | Prompt tokens per step. |

## `fake`

The `fake` family replies with fixed text and loads no model. It is for tests and for trying the server.

| Key | Default | Meaning |
| --- | --- | --- |
| `reply` | `Hello from the fake engine.` | The text of every reply. |
| `token_delay_ms` | 0 | Pause before each token, in milliseconds. |
