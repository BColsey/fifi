<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/fifi-logo-dark.svg">
    <img alt="fifi" src="assets/fifi-logo.svg" width="224">
  </picture>
</p>

# fifi

fifi is a local LLM inference engine and model server written in Rust. It runs
supported models on Apple Silicon using MLX/Metal and on NVIDIA GPUs using CUDA.
It provides OpenAI-style and Anthropic-style HTTP APIs, with streaming and tool
calls.

Read the [getting-started guide](doc/getting-started.md) to build fifi and run
the server. To connect an application, see [Connect a client](doc/how-to/clients.md).

## Models

fifi runs supported checkpoints downloaded from Hugging Face. Depending on the
model and backend, these include MLX, GGUF, EXL3, and NVFP4 formats. Download
the files to local storage before loading them.

Support depends on the model architecture, checkpoint, tokenizer, and backend;
not every model or quantization in these formats is supported. See the
[model reference](doc/reference/models.md) for checkpoint sources and context
limits, and the [current status](doc/explanation/status.md) for qualification
and known limitations.

The model guides cover [Qwen3.6](doc/how-to/qwen36.md),
[Flint](doc/how-to/flint.md), and [Nemotron 3.5 Super](doc/how-to/nemotron.md).

## Configuration

fifi checks memory requirements before loading a model. Supported backends can
also reuse an exact token prefix across requests. See the
[configuration reference](doc/reference/configuration.md),
[example configurations](engine/config/README.md),
[memory planning](doc/explanation/memory.md), and
[conversation reuse](doc/explanation/prefix-reuse.md).

Use `enginesrv doctor`, `check`, and `config effective` to inspect a setup. See
[diagnostics](doc/how-to/diagnostics.md) for commands and troubleshooting.

## Documentation

The [HTTP API reference](doc/reference/http-api.md) describes supported request
fields. The [OpenAPI specification](api/openapi.yaml) describes the endpoints.
The [environment reference](doc/reference/environment.md) lists environment
variables.

See [benchmarks](doc/explanation/benchmarks.md) and
[current status](doc/explanation/status.md) for measured configurations and
results.

## Development

From `engine/`, run the portable checks with:

```sh
cargo fmt --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked
```

These checks do not replace CUDA or Apple Silicon device qualification. Some
tests require external fixtures. Set `FIFI_TEST_REQUIRE_DATA=1` when missing
fixtures must be treated as failures.

For a bug report, include the model, checkpoint, backend, configuration, build
revision, and steps to reproduce it. A reviewed
[diagnostic bundle](doc/how-to/diagnostics.md) can help with startup failures.

## Source code

```sh
git clone https://github.com/zbrdc/fifi.git
```

The server executable is named `enginesrv`. The Rust crate is named
`inference-engine`.

## License

fifi is distributed under the [Apache License 2.0](LICENSE). See
[third-party notices](THIRD_PARTY_NOTICES.md) for third-party code and licenses.
