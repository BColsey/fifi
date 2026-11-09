# HTTP API

`enginesrv` serves the routes below on the `listen` address of its [configuration file](configuration.md).
The contract is in [`api/openapi.yaml`](../../api/openapi.yaml). The completion, token and template routes are extensions that the contract file does not describe.

## Routes

| Method and path | Key needed | Purpose |
| --- | --- | --- |
| `GET /health` | no | Answers `{"status":"ok"}` while the process runs. |
| `GET /ready` | no | `200 {"status":"ready"}` when at least one engine is ready or busy; otherwise `503` with code `not_ready` and the state of each engine. |
| `GET /v1/models` | yes | Model names, each with the engines that serve it and their states. |
| `POST /v1/chat/completions` | yes | OpenAI-style chat completion, with or without streaming. |
| `POST /v1/completions` | yes | Raw text completion of one prompt string, with or without streaming. The chat template is not applied. |
| `POST /v1/messages` | yes | Anthropic-style Messages request, with or without streaming. |
| `POST /v1/messages/count_tokens` | yes | Token count of a Messages request's prompt. No generation runs. |
| `POST /v1/token/encode` | yes | Token ids of a text, or of chat messages after the chat template. No generation runs. |
| `POST /v1/token/decode` | yes | Text of a list of token ids. No generation runs. |
| `POST /v1/apply-template` | yes | The prompt text that the chat template makes of chat messages. No generation runs. |

## Authentication

Send the API key as `Authorization: Bearer <key>`.
The `/v1/messages` routes also accept `x-api-key: <key>`; when that header is present it must match, and the bearer header is not checked.
A missing or wrong key returns `401`.

## Chat completions

Request fields:

| Field | Default | Notes |
| --- | --- | --- |
| `model` | | A name from `/v1/models`. The first engine serving it takes the request. |
| `messages` | required | |
| `stream` | `false` | Server-sent events, ending with `data: [DONE]`. |
| `max_tokens` | 2,048 | Lowered when the prompt plus `max_tokens` would pass the engine's context limit. |
| `temperature` | from the model's `generation_config.json`, else 0.7 | 0 to 2. `0` selects greedy decoding. |
| `top_p` | from `generation_config.json`, else 1.0 | 0 to 1. |
| `top_k` | from `generation_config.json`, else 0 | `0` turns top-k off. |
| `stop` | none | Stop strings, matched on the decoded text. |
| `tools` | none | OpenAI function tools. |
| `tool_choice` | `auto` | `auto` or `none`. `required` and a named function return `400`. |
| `response_format` | `text` | `json_object` and `json_schema` return `400`. |
| `x-thinking` | `default` | Server extension: `on`, `off` or `default` (the chat template decides). |
| `chat_template_kwargs` | none | Only `enable_thinking` is accepted: `true` acts as `x-thinking: on`, `false` as `off`. |
| `x-engine` | none | Server extension: the `id` of the engine to use. Overrides `model`. An unknown id returns `404`. |
| `x-session` | none | Binds to the serving engine on first use and sticks to it; unbound with no other routing field returns `404`. |

The request needs `model` or `x-engine`.

The reply message has `content`, `tool_calls` when the model called a tool, and `reasoning_content` with the thinking text when there was any.
`finish_reason` is `stop`, `length` or `tool_calls`.

`usage` has `prompt_tokens`, `completion_tokens` and `total_tokens`, plus `reused_tokens` and `prompt_tokens_details.cached_tokens`: the prompt tokens served from a saved prefix.
`ENGINESRV_CACHE_USAGE=0` removes the last two.

## Completions

`POST /v1/completions` continues one prompt string.
The prompt is tokenized as written: the chat template is not applied and no start-of-sequence token is added.
Text in the prompt that spells a special token, such as the end-of-turn marker `<|im_end|>`, is tokenized as that special token. The server does not escape it, unlike the content of chat messages.
The reply text is the generated text as it is, with no think block, tool call or whitespace cut out.

| Field | Default | Notes |
| --- | --- | --- |
| `model` | | As for chat completions. |
| `prompt` | required | One non-empty string. A list of strings or of token ids returns `400`. |
| `stream` | `false` | Server-sent events, ending with `data: [DONE]`. |
| `max_tokens` | 2,048 | As for chat completions. |
| `temperature`, `top_p`, `top_k` | as for chat completions | Same bounds and defaults. |
| `stop` | none | One string or an array of strings, matched on the decoded text. The stop text is not part of the reply. |
| `x-engine`, `x-session` | none | As for chat completions. |

`n` other than 1, `echo: true`, and a `logprobs`, `best_of` or `suffix` that is not `null` return `400`.
`seed`, `presence_penalty`, `frequency_penalty`, `logit_bias`, `stream_options` and `user` are accepted and ignored, as on chat completions.
A stream always ends with the usage chunk, whatever `stream_options` says.
A body over 16 MiB returns `400`.

The reply has `object` `text_completion` and one choice with `index`, `text`, `logprobs` (always `null`) and `finish_reason` (`stop` or `length`).
`usage` has the fields of chat completions.
A stream sends one chunk for each piece of text, then a chunk with empty `text`, the `finish_reason` and the `usage`, then `data: [DONE]`.
Chunks carry no role.
A failure after the first chunk sends an `event: error` and no `[DONE]`.
The server marks no prefix boundary for a completion, so it does not save or restore a prefix for it.

## Token and template routes

The three routes below use the tokenizer and chat template of the selected engine.
They queue no job, so they answer while an engine is loading or after it failed.
Each takes `model`, `x-engine` and `x-session` as chat completions do: `x-engine` overrides `model`, an unknown engine or model returns `404`, and a request with none of them returns `400`.
A bad field returns `400` with the field's name in the message, and so does a body over 16 MiB.
These routes read no sampling field: `temperature`, `top_p`, `top_k`, `max_tokens`, `stop` and `stream` are ignored.
Each served request writes one line to the request log, as `POST /v1/messages/count_tokens` does; a failed request writes none.
The line has no finish reason or timings. Its `prompt_tokens` is the number of IDs returned by encode, received by decode, or zero for apply-template.

`POST /v1/token/encode` takes either `text` or `messages`, not both.

| Field | Notes |
| --- | --- |
| `text` | A string, tokenized as written. No start-of-sequence token is added, and the text of a special token such as `<|im_start|>` becomes that token. |
| `messages` | Chat messages, rendered with the chat template and the generation prompt, then tokenized. The ids are the ids that a chat completion with the same fields sends to the engine. |
| `tools`, `tool_choice`, `x-thinking`, `chat_template_kwargs` | With `messages` only. They have the meaning and the checks of chat completions, so `response_format` of type `json_object` or `json_schema` and the `tool_choice` values that chat completions refuse return `400`. |

The reply is `{"tokens": [...], "length": n}`: the ids, and their count.

`POST /v1/token/decode` takes `tokens`, an array of token ids.
Each id is an integer from 0 to the vocabulary size minus 1; an id outside that range returns `400`.
An array longer than the selected engine's context limit, in tokens, returns `400`.
The reply is `{"text": "..."}`. Special tokens are always left out of the text; there is no switch to keep them.

`POST /v1/apply-template` takes `messages` and the same optional fields as encode with `messages`.
The reply is `{"prompt": "..."}`: the text that a chat completion would tokenize, ending with the generation prompt.
A chat template that renders no text returns `400` with the message `rendered prompt is empty` on this route and on encode with `messages`.

## Messages

Request fields follow the Anthropic API: `model`, `max_tokens` and `messages` are required; `system`, `stop_sequences`, `temperature`, `top_p`, `top_k`, `stream`, `tools`, `tool_choice` and `metadata` are optional.
Sampling bounds and defaults are the same as for chat completions.

| Field | Notes |
| --- | --- |
| `thinking` | `{"type": "enabled"}` or `"adaptive"` turns thinking on, `"disabled"` turns it off. Without it, the chat template decides. `budget_tokens` is checked but does not limit thinking. |
| `tool_choice` | `auto` or `none`. `any` and `tool` return `400`. |
| `metadata` | Accepted and ignored. |

Content blocks in the reply are `text`, `thinking` (with an empty `signature`) and `tool_use`.
`usage.cache_read_input_tokens` counts prompt tokens served from a saved prefix, and `input_tokens` counts the rest. `cache_creation_input_tokens` is always 0.

## Errors

Errors have the form `{"error": {"message": "...", "code": "..."}}`.
The Messages routes use the Anthropic error format instead.

| Status | Code | Meaning |
| --- | --- | --- |
| 400 | `bad_request` | The request is malformed or asks for something the server does not support. |
| 400 | `context_length_exceeded` | The prompt does not fit the engine's context limit. |
| 401 | `unauthorized` | Missing or wrong API key. |
| 404 | `not_found` | Unknown route, model or engine. |
| 409 | `memory_plan`, `context_limit` | The engine cannot fit the request in its memory plan or context. |
| 422 | `checkpoint_failure` | The engine's checkpoint failed to load or read. |
| 500 | `backend_error` | The engine failed while running the request, or is in state `error` or `native-failed`. |
| 503 | `overloaded` | The engine's queue is full, or the engine is not available. |
| 503 | `not_ready` | Returned only by `/ready`. |

## Engine states

`/v1/models` and the `not_ready` message report each engine as `loading`, `ready`, `busy`, `error` (it failed to load), `stopped`, or `native-failed` (the GPU backend failed and the engine takes no more requests).

If every engine fails to load, the server does not stay up to report it: the process exits with a non-zero code and prints each engine's cause on standard error.
So `/ready` answers `503` while an engine is still loading. A failure after a successful start shows as `native-failed`, not `error`.
