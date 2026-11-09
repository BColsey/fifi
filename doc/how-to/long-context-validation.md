# Validate long-context recall

Use the long-context driver to check a cold, synthetic prompt at a server-counted
length. It places three random codes near the beginning, middle, and end, then
checks the complete reply. Run it only when the server and device are available
for the test; it does not reserve hardware or verify the running binary's identity.

## Run a 128k prompt

Start a fifi engine with enough context for the prompt and reserved reply. From
the repository root, choose a writable scratch directory and run:

```sh
mkdir -p "$HOME/fifi-scratch/long-context"
export TMPDIR="$HOME/fifi-scratch/long-context"
python3 scripts/bench_long_context.py \
  --url http://127.0.0.1:18792 \
  --model fifi-q36 \
  --prompt-tokens 131072 \
  --max-tokens 8192 \
  --output "$TMPDIR/recall-128k.json"
```

The server's `/v1/system`, `/v1/token/encode`, and `/v1/chat/completions`
endpoints must be available at the supplied URL. `/v1/recent-requests` is
optional and supplies the decode rate when it can be matched to the response.
If the server uses an API key, pass its environment variable name with
`--api-key-env FIFI_API_KEY`; the key is not written to the result.

The driver counts the rendered chat prompt using the server's tokenizer. It
requires at least `--prompt-tokens` and allows up to `--tolerance` additional
tokens, 128 by default. Prompt and reply both consume context. A 131,072-token
prompt with an 8,192-token reply reservation therefore needs at least 139,264
tokens of context, plus any tokens by which the selected prompt exceeds its target.

For a run near a 262,144-token *total* context limit, leave room for a reply:

```sh
python3 scripts/bench_long_context.py \
  --url http://127.0.0.1:18792 \
  --model fifi-q36 \
  --prompt-tokens 257900 \
  --max-tokens 4096 \
  --output "$TMPDIR/recall-near-256k.json"
```

A 262,144-token input plus a reply requires a context limit greater than
262,144. The driver checks both the requested and actual prompt lengths against
the reported limit before sending the generation request.

## Read the result

The JSON result records the counted prompt and completion tokens, prefix reuse,
finish reason, expected codes, answer, and `recall_passed`. A passing recall
requires all three codes in order and a normal `stop` finish. A truncated or
incorrect answer is saved as a completed failure and the command exits with
status 2. A missing token count, unconfirmed cold prompt, or incomplete stream
leaves a `.partial` failure record and exits with status 1. Existing output and
partial files are not overwritten.

`estimated_prompt_tokens_per_second` divides the server-counted prompt by the
time until the first visible streamed content. It includes tokenization,
queueing, and response overhead. When a matching completed server record is
available, `server_prefill_ms` measures engine start to the first generated
batch, and `server_prompt_tokens_per_second` divides the cold prompt count by
that interval. It includes the first-batch work, so it is also an estimate.
`server_queue_wait_ms` and `server_decode_tokens_per_second` come from the same
record when those fields are available. Stream chunks are never counted as
tokens.

This synthetic check does not establish coding quality, multi-turn behavior,
device isolation, or exact build identity. Record the running binary, model,
hardware, and isolation separately when using the result as performance evidence.
