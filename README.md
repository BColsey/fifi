<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/fifi-logo-dark.svg">
    <img alt="fifi" src="assets/fifi-logo.svg" width="224">
  </picture>
</p>

<h1 align="center">fifi</h1>

<p align="center">
  <strong>Local LLM inference in Rust, built around explicit memory budgets and verifiable execution.</strong>
</p>

<p align="center">
  Apple Silicon · NVIDIA CUDA · OpenAI-compatible chat · Anthropic-compatible messages
</p>

<p align="center">
  <a href="#get-started">Get started</a> ·
  <a href="#model-and-hardware-support">Supported models</a> ·
  <a href="#how-fifi-works">Architecture</a> ·
  <a href="#performance-and-qualification">Performance</a> ·
  <a href="doc/getting-started.md">Documentation</a>
</p>

---

fifi is a Rust inference engine and local model server. It executes supported language models through MLX/Metal on Apple Silicon and native CUDA kernels on NVIDIA GPUs. Its enginesrv process exposes OpenAI-style and Anthropic-style APIs, with streaming, tool calls, and multi-engine configuration.

The project concentrates on four areas: explicit memory budgets, conversation-prefix retention, numerically accountable quantized execution, and complete-request measurement.

**Project status:** fifi is under active development. Support is model-, format-, backend-, and configuration-specific. Some model families have working serving paths. Others have source code or device tests without end-to-end qualification. Read the model matrix and [current measured status](doc/explanation/status.md) before treating an experimental path as production-ready.

## What fifi offers

| Capability | What it means in practice |
| --- | --- |
| Native execution | Rust orchestration with MLX/Metal or CUDA model-specific execution paths. |
| Memory admission | fifi budgets weights, KV cache, recurrent state, prefill workspace, and prefix storage before it admits a model. If a requested profile does not fit, startup refuses it rather than silently shrinking context. |
| Conversation reuse | Supported backends can restore an exact token prefix, reducing repeat prompt work on later turns. Support and memory requirements vary by model. |
| Local serving | Authenticated HTTP endpoints for chat, completions, messages, tokenization, and model discovery, with streaming and tool-call responses. A built web frontend can also be served at `/` (`web_dir` in the configuration file). |
| Inspect before loading | `doctor`, `check`, and `config effective` expose hardware capability, preflight refusals, and the origin of resolved settings. |
| Evidence-driven optimization | The project uses numerical references, fixed fixtures, device regressions, and paired benchmarks to evaluate kernel and executor changes. |

fifi loads tested checkpoints with known tokenizer identities, quantizations, and context profiles. It does not accept arbitrary Hugging Face repositories.

## Model and hardware support

The matrix describes the documented serving state of the published repository. Experimental branches are out of scope.

| Model family | Apple Silicon / MLX | NVIDIA / CUDA | Notes |
| --- | --- | --- | --- |
| Qwen3.6-35B-A3B | Serving | Serving | MLX 4-bit checkpoint. The CUDA GGUF build is the documented default. This tree also implements CUDA EXL3 serving, which has separate qualification requirements. |
| Flint | Serving | Pilot | CUDA uses EXL3. Short-context serving works, while long-context numerical quality is not fully qualified. |
| Nemotron 3.5 Super | — | Limited serving | NVFP4 checkpoint with greedy decoding only. Host-assisted execution is substantially slower than a fully GPU-resident model. |
| Qwen3.8-Flash-Next | Source / experimental | Not end-to-end qualified in this published tree | Kernel and model-family development only. This tree ships no approved serving profile for it. |
| Qwen3.8-27B Dense | Source / experimental | Not end-to-end qualified | Not an established deployment target here. |
| Fake test engine | No native backend required | No native backend required | Built-in fixed-response engine for testing the HTTP server and client wiring. |

See [model identities, checkpoint sources, and context limits](doc/reference/models.md), plus the model-specific guides. "Serving" does not imply that every context length, KV format, sampling mode, or multi-device topology is qualified.

## Get started

### 1. Try the server without a GPU or model download

You need Rust (via rustup), Git, and a C compiler. The repository pins its Rust toolchain.

```sh
git clone https://github.com/zbrdc/fifi.git
cd fifi/engine
cargo build --release --bin enginesrv
```

Create `fifi.json` in `engine/`:

```json
{
  "listen": "127.0.0.1:18791",
  "api_key_env": "FIFI_API_KEY",
  "engines": [
    {
      "id": "demo",
      "family": "fake",
      "model_dir": "src/server/testdata/tokenizer",
      "tokenizer_dir": "src/server/testdata/tokenizer"
    }
  ]
}
```

Start the server:

```sh
export FIFI_API_KEY='replace-this-with-a-local-test-key'
target/release/enginesrv --config fifi.json
```

In a second terminal, export the same test key, check readiness, and send a request:

```sh
export FIFI_API_KEY='replace-this-with-a-local-test-key'
curl http://127.0.0.1:18791/ready

curl http://127.0.0.1:18791/v1/chat/completions \
  -H "Authorization: Bearer $FIFI_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"fake","messages":[{"role":"user","content":"Hello"}]}'
```

The fake engine replies with a fixed sentence. This verifies compilation, authentication, routing, and the API shape. It runs no model inference. The [complete walkthrough](doc/getting-started.md) also covers streaming and the Messages endpoint.

### 2. Build for a real GPU

From `fifi/engine`, choose the matching backend:

```sh
# Apple Silicon (macOS): Xcode command-line tools, CMake and MLX build dependencies
cargo build --release --bin enginesrv --features native-mlx

# NVIDIA GPU (Linux): NVIDIA driver and CUDA 12 toolkit
cargo build --release --bin enginesrv --features native-cuda
```

CUDA compilation targets SM 86 by default on x86-64 (RTX 30-series). Set `INFERENCE_CUDA_ARCHS` for other supported architectures, for example `INFERENCE_CUDA_ARCHS=89` for SM 89. A successful build alone does not qualify a model on that GPU.

Recommended first real model: follow [Run Qwen3.6](doc/how-to/qwen36.md), which covers the tested checkpoint and tokenizer files, RTX 3090 and Mac profiles, and memory requirements. There are also guides for [Flint](doc/how-to/flint.md) and [Nemotron 3.5 Super](doc/how-to/nemotron.md).

With a supported Qwen3.6 checkpoint and its tokenizer files already on disk, simplified startup can discover the model:

```sh
export ENGINESRV_API_KEY='replace-this-with-a-secret-key'
target/release/enginesrv serve --model /absolute/path/to/model --print-config
target/release/enginesrv serve --model /absolute/path/to/model
```

`serve --model` uses documented family defaults and reports experimental configurations explicitly. To select a precise context, KV format, prefix-reuse policy, or multiple engines, use an [example JSON configuration](engine/config/README.md):

```sh
# Edit model_dir and tokenizer_dir in the selected file first.
target/release/enginesrv --config config/qwen36-cuda-3090-kv8-prefix.json
```

An arbitrary checkpoint can still be refused at startup. The memory ledger, backend availability, checkpoint identity, and tokenizer validation remain authoritative.

### 3. Connect an existing client

A running engine accepts OpenAI-style requests at `/v1/chat/completions` and Anthropic-style requests at `/v1/messages`. Discover the available model names with the authenticated `/v1/models` endpoint.

```sh
curl http://127.0.0.1:18791/v1/models \
  -H "Authorization: Bearer $ENGINESRV_API_KEY"
```

For OpenAI-compatible SDKs, use `http://127.0.0.1:18791/v1` as the base URL. For Anthropic-compatible SDKs, use `http://127.0.0.1:18791`. Both need the configured API key. See [Connect a client](doc/how-to/clients.md) for Python examples, tool use, thinking controls, and unsupported API fields.

API compatibility is intentionally bounded. For example, forced tool choice and JSON-schema constrained decoding are not supported by the documented serving API. For supported fields and behavior, refer to the [HTTP API contract](doc/reference/http-api.md).

## How fifi works

```text
OpenAI / Anthropic client
          │
          ▼
       enginesrv ─────── authentication, streaming, request handling
          │
          ▼
   engine / scheduler ── admission, cancellation, sessions, prefix reuse
          │
          ▼
    model executor ───── model-specific weights, routing, attention,
          │             recurrent state, decoding and KV lifecycle
     ┌────┴────┐
     ▼         ▼
 MLX / Metal  CUDA ────── native kernels, quantized projections,
                           GPU/host memory management
```

This is a conceptual map. Model-specific kernels and execution semantics vary.

fifi budgets model weights, KV state, recurrent state, prefill buffers, and cache capacity against available memory before a model runs. Some paths can keep expert weights partly in host RAM. Others require stricter GPU residency. Model weights must be stored locally, not on NFS or another network filesystem. Startup refuses network-mounted model paths. See [Memory planning](doc/explanation/memory.md) and [Configuration / local storage](doc/reference/configuration.md#local-storage).

Prefix reuse is exact-token reuse. A follow-up request can restore compatible saved model state for the unchanged portion of a conversation, subject to the model's backend and cache rules. Saved state comes from the engine itself. External KV snapshots are not accepted. See [Conversation reuse](doc/explanation/prefix-reuse.md).

Multi-GPU execution remains configuration-specific. The CardStack multi-device path is under active development. A passing device-specific test does not imply that every topology, peer-transfer route, or multi-GPU deployment is qualified.

fifi's performance work measures complete request behavior, including routing, memory transfers, synchronization, prompt processing, and quality gates. Isolated GEMM/kernel timing explains those results. See [Benchmarks](doc/explanation/benchmarks.md) and the [dated measurement record](doc/explanation/status.md).

## Diagnose a setup before running it

```sh
# What this build and machine can support
target/release/enginesrv doctor

# Model directory, format, tokenizer and memory preflight
target/release/enginesrv check /absolute/path/to/model --family qwen3_6_35b_a3b

# Validate a complete configuration and inspect resolved settings
target/release/enginesrv config validate config/qwen36-cuda-3090-kv8-prefix.json
target/release/enginesrv config effective config/qwen36-cuda-3090-kv8-prefix.json
```

`doctor --bundle <file>` produces a sanitized JSON diagnostic bundle that can accompany a bug report. The bundle intentionally excludes prompts, completions and API keys. Inspect it before sharing. See [Diagnostics](doc/how-to/diagnostics.md).

If startup refuses a model, do not work around the refusal by assuming more context, a different KV format, or different checkpoint bytes are equivalent. Start with the relevant [model guide](doc/reference/models.md) and effective settings.

## Performance and qualification

fifi's numbers depend on the exact model, checkpoint, quantization, context, cache format, prompt, generation length, hardware, and build. The project measures and reports short single-request throughput, long-context serving, and multi-turn replay separately. Numerical kernel tests and model-level quality gates are separate checks.

The project maintains separate evidence for:

- Functional correctness: build and portable tests, host-side numerical references, API contracts and failure behavior.
- Device correctness: backend-specific and real-weight tests, including quantization, attention, routing, KV lifecycle and fallbacks.
- Quality: independent model-level comparisons and frozen thresholds that require a path to be non-inferior.
- Performance: paired end-to-end runs with stated workloads, plus kernel measurements that explain the serving results.

The [benchmarks page](doc/explanation/benchmarks.md) contains matched-engine comparisons. The [status page](doc/explanation/status.md) records dated receipts, context admissions, limitations and experimental results. Both cover only the configurations they document.

## Development and testing

From `engine/`:

```sh
cargo fmt --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked
```

The repository's GitHub Actions checks cover the portable surface. They do not run full CUDA or Apple Silicon device qualification. Some hardware tests are explicitly ignored until invoked on the appropriate host with the required real model and fixture data.

A fixture-dependent test may skip when its external data is absent. Set `FIFI_TEST_REQUIRE_DATA=1` on a qualification machine where the fixture must be present, so absence becomes a failure rather than an apparent success. See the [getting-started guide](doc/getting-started.md) and the [source/configuration references](doc/reference/configuration.md) for test paths and setup.

For contributions and issue reports, identify the model family, exact weight pack, backend, context and KV profile, build revision, and a minimal reproduction. For startup failures, attach a reviewed [diagnostic bundle](doc/how-to/diagnostics.md).

## Documentation

| Start here | Reference and deeper reading |
| --- | --- |
| [Getting started](doc/getting-started.md) | [Models and checkpoints](doc/reference/models.md) |
| [Run Qwen3.6](doc/how-to/qwen36.md) | [Configuration](doc/reference/configuration.md) |
| [Run Flint](doc/how-to/flint.md) | [Environment variables](doc/reference/environment.md) |
| [Run Nemotron](doc/how-to/nemotron.md) | [HTTP API](doc/reference/http-api.md) · [OpenAPI](api/openapi.yaml) |
| [Connect a client](doc/how-to/clients.md) | [Memory planning](doc/explanation/memory.md) |
| [Troubleshooting / diagnostics](doc/how-to/diagnostics.md) | [Prefix reuse](doc/explanation/prefix-reuse.md) |
| [Example configs](engine/config/README.md) | [Benchmarks](doc/explanation/benchmarks.md) · [Current status](doc/explanation/status.md) |

## License

fifi is licensed under Apache License 2.0. [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) documents third-party code and its licenses.

The executable is `enginesrv`. The Rust crate is currently named `inference-engine`.
