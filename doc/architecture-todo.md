# Shared architecture TODO

Share decisions that must agree across serving and diagnostics. Keep model-specific execution order, numerical paths, and device ownership with their families.

- [x] **Share `gemma4` build eligibility.** The family name is recognized on every build, but the starter only serves Nemotron Bolt E4B on CUDA. Startup rejects a build without CUDA before reading sampling defaults. `check` reports it as unsupported for both explicitly named and GGUF-detected entries.
- [ ] **Compare checkpoint eligibility across boot, simplified startup, and preflight.** Use the same model-identity rules where applicable without merging their discovery rules or treating an eligible checkpoint as qualified.
- [ ] **Publish settings from the engine that opened.** Flint's server wrapper reads a process-wide prefill order immediately after starting an asynchronous load. Publish the actual order per engine when its worker becomes ready. Test delayed success, the jobs-to-token-major fallback, and isolation between engine IDs.
- [ ] **Preflight the options from each support-bundle entry.** The bundle still projects config entries onto partial candidate options. Start with non-default Flint options. Keep estimated fit, actual admission, and quality qualification separate; do not initialize a device from the serving diagnostics endpoint.
- [ ] **Test scheduler invariants across single and batched workers.** Upstream now attributes batched expert-bank counters per piece. Keep that behavior and test cancellation, denied-prefill retries, prefix resume, and publication through both workers before sharing more lifecycle code.
- [ ] **Qualify shared EXL3 prefill before extending it.** Compare Flash-Next's opt-in and legacy paths on chunk tails, prefix resume, memory admission, and output quality. Keep family-owned launch ordering and existing defaults unless qualification supports a change.

See [model support](reference/models.md) and [diagnostic commands](how-to/diagnostics.md).
