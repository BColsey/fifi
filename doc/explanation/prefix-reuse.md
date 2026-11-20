# How conversation reuse works

Chat clients send the whole conversation with every request.
Without reuse, the engine processes every token of that history again on each turn, so replies take longer to start as the conversation grows.

With `"prefix_reuse": true` in the configuration file, the engine saves its state at the end of each finished turn. An omitted key resolves per family at startup: on for the qualified reuse paths (Flash-Next CUDA and Qwen3.6 CUDA on the CUDA build, since the 2026-11-13 qualification), off for every other family - the boundary marking alone would only split their prefill and change their outputs.
When the next request starts with the same tokens, the engine restores that state and processes only the new part.
On a long conversation this turns tens of seconds of prompt processing into a few seconds.

## Matching the prompt

Reuse needs the new prompt to start with exactly the tokens the engine saw.
Three settings help with that:

- **Warm turns** (`ENGINESRV_WARM_TURNS`, on by default). The server remembers the token sequence of the last conversation, so the next turn only tokenizes its new messages. A request from another conversation replaces that memory; the seeded slot is kept for an identical retry (`ENGINESRV_SEED_RESENDS`, on by default; any set value other than `1`, `true` or `on` disables it), and only with that switch also on, a per-engine `ENGINESRV_RESEND_EXTRA_BYTES` budget above zero additionally retains one displaced conversation's completed result — the budget alone retains nothing.
- **Block saves** (`ENGINESRV_PREFIX_BLOCK_MATCH`, on by default). Besides the end of each turn, the engine saves one block checkpoint, cut at a 4,096-token boundary just before the end of the prompt. On Qwen3.6 with CUDA each conversation keeps its own block checkpoint: a later cut of the same conversation replaces its earlier one, and the checkpoints of other conversations stay until the store's entry limit evicts the least recently used. Other engines keep one block checkpoint per numeric path. The checkpoint lands up to 4,096 tokens before the turn's own checkpoint, and block saves need a store that keeps at least two saved prefixes. When a client rewrites the end of the conversation, the engine restores the block checkpoint before the change instead of starting over. In one measurement, a 174,829-token prompt whose last 23 tokens changed reached its first token in 11.8 s with block saves and 40.7 s without.
- **Reasoning replay** (`ENGINESRV_THINK_REPLAY`, on by default for Qwen3.6). Clients resend the model's earlier reasoning. fifi passes it to the chat template, which keeps the reasoning of turns in the current tool-calling loop. A system message inside that loop is rendered as a tool response (`ENGINESRV_NOTICE_TOOLWRAP`), so the template does not treat it as a new user turn and drop the loop's reasoning.

## Where saved prefixes live

On Apple Silicon, saved prefixes share `session_cap_bytes` with the live session. The store keeps the 6 most recently used prefixes (`prefix_store_entries`).

On NVIDIA CUDA, the newest saved prefix stays on the GPU until a later request overwrites it, so restoring it needs no copy from host memory.
Up to eight saved prefixes are also kept in pinned host memory; see [How fifi plans memory](memory.md#on-nvidia-cuda).
With the disk cache (see the [configuration reference](../reference/configuration.md#prefix-cache-on-disk-cuda)), Qwen3.6 on CUDA also keeps saved prefixes in a directory, but only when pinned memory is not enough: entries about to be evicted are written in the background, and a graceful stop writes the tips that exist only in memory. A prompt that pinned memory cannot serve is looked up there before it is processed again, so a restart or an eviction from pinned memory costs one verified read of the saved state instead of the whole prompt.

## Which engines support it

| Model and backend | Prefix reuse |
| --- | --- |
| Qwen3.6, Apple Silicon | yes; off in the shipped Mac file |
| Qwen3.6, CUDA | yes; on in all shipped CUDA files |
| Flint, Apple Silicon | experimental: also set `FIFI_FLINT_PREFIX_REUSE=1` |
| Flint, CUDA | experimental: also set `FIFI_FLINT_PREFIX_REUSE=1`; `kv_profile` `f16` only |

## Seeing it work

The `usage` of each reply reports how many prompt tokens came from a saved prefix: `reused_tokens` and `prompt_tokens_details.cached_tokens` on the OpenAI API, `cache_read_input_tokens` on the Anthropic API.

The request-log line and `/v1/recent-requests` also carry the engine's own
prefix report: `prefix_entries` and `prefix_bytes`, the stored snapshots the
executor holds for the request's engine. On Qwen3.6 with CUDA,
`prefix_bytes` is the committed payload of those entries. The RAM the prefix
store actually holds is larger: pinned slots are allocated at capacity, so a
half-full entry still occupies its full slot. Read `prefix_bytes` as the
payload figure, not as the store's memory footprint.
