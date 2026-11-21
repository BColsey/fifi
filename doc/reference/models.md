# Models

| `family` | Model | Apple Silicon | NVIDIA CUDA | Checkpoint |
| --- | --- | --- | --- | --- |
| `qwen3_6_35b_a3b` | Qwen3.6-35B-A3B | yes | yes | MLX: `mlx-community/Qwen3.6-35B-A3B-4bit`. CUDA: `Qwen3.6-35B-A3B-Q4_K_M.gguf` from `ggml-org/Qwen3.6-35B-A3B-GGUF`, or an [EXL3 pack](configuration.md#exl3-pack-cuda). |
| `flint` | Flint, 39.8B total, 2.5B active | yes | pilot | MLX: `RecursiveSI/flint-int4`. CUDA: `0xBakeer/TandemLLM-Flint-EXL3-4.0`. |
| `nemotron35` | Nemotron 3.5 Super, 120B total, 12B active | no | yes, greedy only | `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4` at revision `8a115b9d`. |
| `qwen3_8_flash_next` | Qwen3.8-Flash-Next | in the source tree | opt-in | Not validated end to end; A `native-cuda` Linux build serves it on the card unless `flash_next.cuda_decode` is false. |
| `qwen3_8_27b_dense` | Qwen3.8-27B | in the source tree | no | Not validated end to end. |
| `inkling2_loop` | Inkling 2 Mini | CPU pilot | opt-in | Native ModelOpt safetensors and tokenizer files from the [pinned release](https://huggingface.co/thinkingmachines/Inkling-2-Mini/tree/ed40e7da625f594fc9c4990f5f66be24873c22d7). |
| `gemma4` | Gemma4 | staged | staged | Registered name only: a config naming it starts no engine ("gemma4 is staged in this build"). |
| `fake` | none | yes | yes | Test engine; replies with fixed text. |

"Pilot" means the model serves short contexts and its long-context quality is not measured.

On CUDA, any `.gguf` conversion of Qwen3.6-35B-A3B in `model_dir` serves (the largest when there are several; `Q36_ALLOW_UNPINNED_GGUF=0` serves only the pinned file); the strict open (architecture metadata and full tensor set) is the gate, and the pinned Q4_K_M file stays the qualified checkpoint and wins whenever it is present.

For CUDA GGUF, `qwen36.gguf_sha256` binds startup to an expected full-file digest, including an alternate checkpoint admitted by that switch.
See [content pinning](configuration.md#gguf-content-pinning-cuda).
A content pin does not qualify another quantization or checkpoint for quality.

## Inkling 2 Mini

On a 64-bit build, set `FIFI_INKLING_CPU_REFERENCE=1` and choose family `inkling2_loop` in the `enginesrv` configuration. Use the pinned release directory for `model_dir` and `tokenizer_dir`. On a 64-bit Linux build with `native-cuda`, set `FIFI_INKLING_CUDA_REFERENCE=1` instead to run the BF16 and packed expert projections on the first visible CUDA device. Set `CUDA_VISIBLE_DEVICES` to choose the card. Set only one Inkling backend flag. Both pilots support raw completions and plain-text chat, with one active sequence and at most 128 tokens including the prompt. Routing, attention, KV and convolution state run on the host.

`enginesrv check` inspects the safetensors inventory and reports the BF16 KV storage plan for the fixed loop. With a pilot enabled it also checks the pinned tokenizer and the short-context settings. CUDA admission checks available device memory at startup.

For the CUDA pilot, `FIFI_INKLING_CUDA_RESIDENT_EXPERT_MIB` optionally reserves a budget for packed expert banks. The loader places complete banks for the repeated core layers first and includes their bytes in device admission. A budget of zero uses streamed experts. Use `FIFI_INKLING_CUDA_PROFILE=1` to log per-token time by projection and upload phase.

## Tokenizer files

Each engine reads `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` and `generation_config.json` from its `tokenizer_dir`.
For the three validated families, startup compares the files with the SHA-256 hashes of the tested release and refuses others.
For Qwen3.6 CUDA, use the tokenizer files from `mlx-community/Qwen3.6-35B-A3B-4bit` with either the GGUF file or the EXL3 pack. The other validated checkpoints listed above include matching files.

## Weight formats

| Format | Used by |
| --- | --- |
| MLX affine-quantized safetensors; the quantization is read from `config.json` | Qwen3.6 and Flint on Apple Silicon |
| GGUF with F16, Q4_K, Q5_0, Q6_K and Q8_0 tensors | Qwen3.6 on CUDA |
| EXL3 | Qwen3.6 and Flint on CUDA |
| NVFP4 safetensors | Nemotron 3.5 Super |
| Mixed NVFP4 and MXFP8 safetensors | Inkling 2 Mini |

## Context limits

| Model and backend | Default | Largest |
| --- | ---: | ---: |
| Qwen3.6, Apple Silicon | 262,144 | 262,144 tested on a 36 GB Mac |
| Qwen3.6, CUDA | 32,768 | 262,144 tested on a 24 GB RTX 3090 |
| Flint, Apple Silicon | 32,768 | 32,768 |
| Flint, CUDA | 131,072 | 262,144 |
| Nemotron 3.5 Super | set in config | |
