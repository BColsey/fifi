<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/fifi-logo-dark.svg">
    <img alt="fifi" src="assets/fifi-logo.svg" width="254">
  </picture>
</p>

<p align="center">
  <b>LLM inference in Rust.</b><br>
  Apple Silicon (MLX/Metal) · NVIDIA CUDA
</p>

<p align="center">
  <a href="doc/getting-started.md">Getting started</a> ·
  <a href="#documentation">Docs</a> ·
  <a href="doc/reference/models.md">Models</a> ·
  <a href="doc/reference/http-api.md">API</a>
</p>

---

fifi is an LLM inference engine written in Rust.
It runs models on Apple Silicon through MLX and Metal, and on NVIDIA GPUs through CUDA.
Its server, `enginesrv`, speaks the OpenAI chat completions API and the Anthropic Messages API (`/v1/chat/completions`, `/v1/messages`), with streaming and tool calls.
One process can host several models.

## Models

| Model | Apple Silicon | NVIDIA CUDA |
| --- | --- | --- |
| Qwen3.6-35B-A3B | yes | yes |
| Flint | yes | pilot |
| Nemotron 3.5 Super | | yes |

See [Models](doc/reference/models.md) for checkpoints and context limits.

## Quick start

New to fifi? The [tutorial](doc/getting-started.md) runs the server with a test engine, without a GPU or a model download.

Build with the backend for your machine:

```sh
git clone https://github.com/zbrdc/fifi.git
cd fifi/engine
cargo build --release --bin enginesrv --features native-mlx   # Apple Silicon
cargo build --release --bin enginesrv --features native-cuda  # NVIDIA GPU, Linux
```

Pick a file from [engine/config](engine/config/README.md), set its `model_dir` and `tokenizer_dir`, and start the server:

```sh
export ENGINESRV_API_KEY=my-secret-key
target/release/enginesrv --config config/qwen36-mac-studio-36gb.json
```

Before starting, `enginesrv doctor` reports what this build can serve on this machine (compiled backends, free GPU memory, host RAM), `enginesrv check <model_dir> --family qwen3_6_35b_a3b` preflights one model directory against it, and `enginesrv config validate <file>` and `enginesrv config effective <file>` check a configuration and show the resolved values with their origins. `enginesrv --help` lists every command. To serve the web frontend too, build it (`npm run build` in `web/`) and set `web_dir` in the configuration file; it is then served at `/`.

It listens on `http://127.0.0.1:18791`. Send the key as `Authorization: Bearer <key>`:

```sh
curl http://127.0.0.1:18791/v1/chat/completions \
  -H "Authorization: Bearer $ENGINESRV_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "fifi-q36", "messages": [{"role": "user", "content": "Hello!"}]}'
```

Run the tests with `cargo test`.
A few tests read data that is generated outside Git, such as parity fixtures.
They skip when that data is absent, and they fail instead when `FIFI_TEST_REQUIRE_DATA` is set to exactly `1`; any other value leaves them skipping.
A skipping test still reports `ok`. Its skip message appears only when you run `cargo test -- --nocapture`.
Set it on any host that is supposed to have the data, so a gate cannot pass without running those tests.

Tests that need model weights or recorded data on your machine find them by name.
Each name is an environment variable, or else a key in the JSON object `~/.config/fifi/test-paths.json`, which lives outside the repository so every worktree shares it.
The names are `Q36_CUDA_MODEL_DIR`, `FIFI_QWEN36_MODEL_DIR` (a Qwen3.6 model directory), `Q36_CUDA_COHORT_DIR` (the recorded prompt cohort), `FLINT_P01_GGUF_LOGITS` (a cached logits file) and `FIFI_SCRATCH_ROOT` (the directory under which tests may write).
`M3_LAYOUT_DIR` (the frozen layout directory of the EXL3-versus-GGUF quality comparison) is a further name; the two default tests that read it skip when it is unset.
The ignored `m3_compare` test scores two recorded sidecar sets: set `M3_COMPARE` to `P1:<candidate dir>:<reference dir>` or `O1:<candidate dir>:<reference dir>`, and run `cargo test --lib m3_compare -- --ignored --nocapture` from `engine/`.
An `#[ignore]`d test that needs one of them fails with a message naming it when it is unset; the Qwen3.6 tokenizer tests skip instead.
`FIFI_SCRATCH_ROOT` is also read as an environment variable only, never from the file, by the macOS native qualification code outside test builds, so set it there.

The Git hooks in `.githooks/` are opt-in. Enable them in your clone with `git config --local core.hooksPath .githooks`.
They need `python3`, and they fail if `scripts/check_staged.py` is missing rather than skipping the checks.
GitHub Actions runs the formatting, Clippy, CPU-only test and script checks on every push to `main` and every pull request.
The GPU and Mac qualification runs are not part of that workflow.

## Documentation

**Tutorial**

- [Getting started](doc/getting-started.md)

**How-to guides**

- [Run Qwen3.6](doc/how-to/qwen36.md)
- [Run Flint](doc/how-to/flint.md)
- [Run Nemotron 3.5 Super](doc/how-to/nemotron.md)
- [Connect a client](doc/how-to/clients.md)

**Reference**

- [Configuration file](doc/reference/configuration.md)
- [Environment variables](doc/reference/environment.md)
- [HTTP API](doc/reference/http-api.md) and the [OpenAPI contract](api/openapi.yaml)
- [Models](doc/reference/models.md)

**Explanation**

- [How fifi plans memory](doc/explanation/memory.md)
- [How conversation reuse works](doc/explanation/prefix-reuse.md)
- [Benchmarks](doc/explanation/benchmarks.md)
- [Current status](doc/explanation/status.md)

## License

fifi is licensed under the [Apache License 2.0](LICENSE).
Third-party code and its licences are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

The Rust crate is still named `inference-engine`.
