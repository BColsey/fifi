# KV cache quality at long context

Quantizing the attention KV cache saves memory, but can change the next-token
probabilities used by the model. A context that fits in memory is not by itself
a quality result. [How fifi plans memory](memory.md) gives the cache sizes;
[Current status](status.md) records measured results and their receipts.

This page records the evidence and open questions for a Qwen3.6 study. Flint's
existing evidence is included for later reference; it is not part of that study.

## Qwen3.6

The CUDA `f32` KV setting stores the cache as FP16; `q8` uses an 8-bit cache.
The Apple Silicon `q8` cache passed its own quality gate, but that does not
qualify the CUDA path, a different weight pack, or a new workload. See the
[configuration reference](../reference/configuration.md) for the settings and
[the Qwen3.6 guide](../how-to/qwen36.md) for shipped profiles.

The existing Qwen CUDA KV8 cohort tests compare chunked prefill with a
one-token path **using KV8 in both arms**. Their deep cases include a 65,536- or
196,608-token prefix and 1,024 scored rows. They assess execution paths, not
the effect of q8 versus FP16 KV on task quality. A test in the source is not a
receipt proving that a particular case was run. The [Qwen EXL3 configuration
notes](../reference/configuration.md#exl3-pack-cuda) also describe the current
chunked-prefill evidence and its limits.

The Qwen long-context test can build prompts without a model and compare saved
reports on a CPU. Its model-running implementation is Apple Silicon only. The
CUDA step gate accepts FP16 and q8 KV, feeds an identical forced continuation,
and reports each target logit, log-sum-exp, argmax and top 32. Those summaries
can characterize target-token negative log-likelihood and top-1, but cannot
reconstruct exact full-vocabulary KL. The gate computes a full-row finiteness
flag without publishing or enforcing it, so its raw report alone does not prove
every logit finite. It loads GGUF weights and processes the prompt one token at
a time; results do not qualify EXL3 or a different prefill path.

## Comparing profiles

A useful long-context comparison needs the same token IDs and continuation,
model pack, tokenizer, context capacity and prefill schedule in each arm.
Record the binary, configuration and input identities, actual activated paths,
memory admission and output artifacts. Reserve space for generated tokens: the
CUDA step gate allocates the longest prompt plus 128 scored positions.

Use fresh, held-out coding and general-document examples at the depths of
interest, including copy and retrieval tasks. Report task quality and
per-position diagnostics alongside first-token latency, decode speed and peak
memory. A single successful recall, a successful boot, or matching text alone
does not establish non-inferiority. Check that the chosen cohort and quality
registration apply to the actual model and backend before calling a result
qualified. Lower-bit and mixed-precision formats need their own quality and
admission evidence.

## Flint evidence for later

Flint CUDA offers `f16`, experimental `q8` and opt-in `q8k`, which quantizes
only keys. `q8k` has no quality verdict and does not support prefix reuse. On
8,448 teacher-forced repository-source positions through 8,192 tokens, the
`q8` legacy and tiled paths had mean KL against `f16` of 0.254 and 0.266 nats;
that comparison did not establish equivalence. These are one model and one
source-code domain, not a deep-context qualification. See the
[`q8` quality disclosure](../reference/environment.md#flint) and
[Flint results](status.md#flint-on-an-rtx-3090-cuda) for conditions and receipts.

A separate 32K investigation found that quantizing later layers changed more
argmax tokens than quantizing earlier layers. That observation is diagnostic;
it neither qualifies a mixed-layer format nor establishes a task-quality
failure. The `f16` jobs prefill is the default under the conditions in the
[Flint guide](../how-to/flint.md); its registered 32K quality check failed and
the default rests on an explicit owner override. For a future KV comparison,
hold prefill order constant: the default `f16` jobs path and the token-major
`q8`/`q8k` paths would otherwise confound the result.
