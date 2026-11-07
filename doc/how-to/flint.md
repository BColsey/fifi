# Run Flint

Flint is a mixture-of-experts model with 39.8B parameters, 2.5B of them active per token.
fifi serves it on Apple Silicon. On NVIDIA GPUs it runs as a pilot: short contexts work, and long-context quality is not measured.

If you have not built fifi before, do the [tutorial](../getting-started.md) first.
Run Flint in a server process of its own: it sets the MLX allocator cache limit to 64 MiB for the whole process (rollback: `FIFI_FLINT_MLX_CACHE_LIMIT=0`).

## On a Mac

You need Apple Silicon, Xcode with its command-line tools, CMake, and network access during the first build.
The checkpoint is 22.6 GB.

1. Download the checkpoint:

   ```sh
   hf download RecursiveSI/flint-int4 --revision 3b93f9053b530488405ae9fca51d4900d6ce21f0 \
     --local-dir ~/models/flint-int4
   ```

2. Build with the MLX backend:

   ```sh
   cd fifi/engine
   cargo build --release --bin enginesrv --features native-mlx
   ```

3. Create `flint.json`. Write the paths out in full:

   ```json
   {
     "listen": "127.0.0.1:18791",
     "api_key_env": "ENGINESRV_API_KEY",
     "engines": [
       {
         "id": "flint",
         "family": "flint",
         "model_dir": "/Users/you/models/flint-int4",
         "tokenizer_dir": "/Users/you/models/flint-int4"
       }
     ]
   }
   ```

   Do not add a `flint` object on the Mac; startup refuses one.

4. Start the server and wait for `/ready`:

   ```sh
   export ENGINESRV_API_KEY=my-secret-key
   target/release/enginesrv --config flint.json
   ```

Send requests with `"model": "flint"`.
The context limit is 32,768 tokens. To use less memory, set `FIFI_FLINT_MLX_CONTEXT_LIMIT=8192` before starting.

## On an NVIDIA GPU (pilot)

You need Linux with the CUDA 12 toolkit.
The pack is 20.7 GB.
Up to a context of about 60,000 tokens (`f16`, token-major prefill; the default job-table prefill reserves about 0.65 GB more, so slightly less) every weight stays on the GPU.
A longer context, which is the default, keeps part of the experts in about 18 GiB of page-locked host memory, so the machine needs that much free memory on top of the GPU.

1. Download the EXL3 pack. It includes the tokenizer files:

   ```sh
   hf download 0xBakeer/TandemLLM-Flint-EXL3-4.0 --revision e7ab11242b0b30a6063f3559dda75fed6b9e12c6 \
     --local-dir ~/models/flint-exl3
   ```

2. Build with the CUDA backend:

   ```sh
   cd fifi/engine
   cargo build --release --bin enginesrv --features native-cuda
   ```

3. Create `flint-cuda.json`. On CUDA the `flint` object is optional: without it the engine serves a context of 131,072 tokens with the `f16` KV cache. To choose the context limit and the KV cache format, add it:

   ```json
   {
     "listen": "127.0.0.1:18791",
     "api_key_env": "ENGINESRV_API_KEY",
     "engines": [
       {
         "id": "flint",
         "family": "flint",
         "model_dir": "/home/you/models/flint-exl3",
         "tokenizer_dir": "/home/you/models/flint-exl3",
         "flint": {"context_limit": 4096, "kv_profile": "f16"}
       }
     ]
   }
   ```

4. Start the server as above and wait for `/ready`.

**NOT MERGED: the registered gate FAILED on H0 of the v6 run, so this default was not taken; the text describes branch task/flint-jobs-default4 only.** On the `f16` KV profile the engine prefills a prompt with the job-table layer-major pass by default, with a hi/lo split of the activations of its tall projections that removes the F16 rounding they would otherwise add. The first token of a 16,000-token prompt falls from 118 to 21 s (4,000 tokens: 28 to 4.7 s), and the first request after a prefill decodes somewhat slower while the expert bank refills (about 13 percent at 16K, level on the next request). The registered 32K quality check of the plain pass fails two of four bars and covers a jobs prefill of only 11,519 rows; the split's own registered check failed on its default-plan draw, and its evidence is cited at `FIFI_EXL3_LAYER_MAJOR` and `FIFI_EXL3_TALL_HILO` in the environment reference, and the default applies to prompts up to the full context. The start line `Flint prefill order: ...` says which order the engine took and why, and the load line carries `prefill_order_source=default|explicit` and `tall_hilo=`. The engine keeps the token-major walk when the profile is not `f16`, when CPU experts are configured, when `prefill_chunk_size` is not 8, or when `FIFI_FLINT_CUDA_ALLOC_CACHE=0`; if the job-table pass does not fit in memory it retries once on the token-major walk and says so. To roll back, set `FIFI_EXL3_LAYER_MAJOR=off` and restart; to keep the pass without the split, set `FIFI_EXL3_TALL_HILO=0`. Prefix reuse under the job-table pass has not been run on a GPU yet; a prefix saved under one order or split state is never restored under another.

Startup checks free GPU memory and free host memory and refuses a load that does not fit; lower `context_limit` if it is refused.
The load line says `expert_slots=` and `bank=auto` when part of the experts is in host memory. To keep every expert on the GPU, set `FIFI_FLINT_CUDA_EXPERT_CACHE=0`; a context that does not fit then is refused with the largest one that does.
Prefix reuse is experimental on CUDA: set `FIFI_FLINT_PREFIX_REUSE=1` and `"prefix_reuse": true`.
It needs `"kv_profile": "f16"`; with `q8` the engine logs a note and serves without reuse.
Greedy decoding and top-k sampling with k up to 20 run on the GPU; other sampling settings run on the host and are slower.

If the model produces a non-finite value, that request fails with `500 backend_error` and the engine keeps serving.
A CUDA runtime error stops the engine; restart the server.

## See also

- [`flint` options](../reference/configuration.md#flint) and [Flint environment variables](../reference/environment.md#flint)
- [How fifi plans memory](../explanation/memory.md)
