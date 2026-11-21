#!/usr/bin/env python3
"""Exercise an isolated enginesrv with the original Inkling 2 Mini checkpoint."""

import argparse
from contextlib import contextmanager
import http.client
import json
import os
from pathlib import Path
import time
from urllib.parse import urlencode, urlsplit

RAW_PATH = "/v1/completions"
CHAT_PATH = "/v1/chat/completions"
RAW_PROMPT = "The capital of France is"
# Common three-token prefix in native and llama.cpp UD-IQ2_XXS reference text,
# encoded with the publisher's pinned tokenizer.
RAW_PREFIX = " Paris.\n\n"
RAW_PREFIX_IDS = [152717, 13, 68014]
RAW_PROMPT_IDS = [371, 151451, 149879, 151194, 149884]
EOS_ID = 200006
THINKING_OFF = {"enable_thinking": False}


def require(condition, message):
    if not condition:
        raise ValueError(message)


class Server:
    def __init__(self, url, key, timeout):
        self.url = url
        self.key = key
        self.timeout = timeout

    @contextmanager
    def request(self, path, body=None, timeout=None):
        connection_type = (http.client.HTTPSConnection if self.url.scheme == "https"
                           else http.client.HTTPConnection)
        connection = connection_type(self.url.netloc, timeout=timeout or self.timeout)
        headers = {"Authorization": f"Bearer {self.key}", "Connection": "close"}
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        try:
            connection.request("POST" if body is not None else "GET", path, data, headers)
            with connection.getresponse() as response:
                if response.status != 200:
                    detail = response.read(4096).decode(errors="replace")
                    raise ValueError(f"{path} returned HTTP {response.status}: {detail}")
                yield response
        finally:
            connection.close()

    def json(self, path, body=None, timeout=None):
        with self.request(path, body, timeout) as response:
            return json.load(response)


def events(response):
    """Read complete SSE events, including multiline data, and ignore keepalives."""
    parts = []
    for line in response:
        line = line.decode("utf-8").rstrip("\r\n")
        if not line:
            if parts:
                yield "\n".join(parts)
                parts.clear()
        elif line.startswith("data:"):
            parts.append(line[5:].removeprefix(" "))
    require(not parts, "unterminated SSE event")


def choice_of(packet):
    choices = packet["choices"]
    require(len(choices) == 1 and choices[0]["index"] == 0, "expected one choice at index 0")
    return choices[0]


def normalize(reply, chat):
    choice = choice_of(reply)
    message = choice["message"] if chat else {}
    if chat:
        require(message["role"] == "assistant", "expected an assistant message")
    return {
        "text": (message.get("content") or "") if chat else choice["text"],
        "reasoning": (message.get("reasoning_content") or "") if chat else "",
        "finish": choice["finish_reason"],
        "usage": {key: reply["usage"][key]
                  for key in ("prompt_tokens", "completion_tokens", "total_tokens")},
    }


def fold_stream(frames, chat):
    text, reasoning = [], []
    reply_id = None
    finish = None
    assistant_role = False
    done = False
    for frame in frames:
        require(not done, "SSE event after [DONE]")
        if frame == "[DONE]":
            require(finish is not None, "[DONE] before finish and usage")
            done = True
            continue
        packet = json.loads(frame)
        require("error" not in packet, f"stream error: {packet}")
        require(finish is None, "SSE chunk after terminal chunk")
        reply_id = reply_id or packet["id"]
        require(packet["id"] == reply_id, "SSE reply ID changed")
        require(packet["object"] == ("chat.completion.chunk" if chat else "text_completion"),
                "wrong SSE chunk shape")
        choice = choice_of(packet)
        if chat:
            delta = choice["delta"]
            if not assistant_role:
                require(delta.get("role") == "assistant", "missing initial assistant role")
                assistant_role = True
            else:
                require("role" not in delta, "unexpected repeated role")
            text.append(delta.get("content") or "")
            reasoning.append(delta.get("reasoning_content") or "")
        else:
            text.append(choice["text"])
        if choice.get("finish_reason") is not None:
            finish = choice["finish_reason"]
            usage = packet["usage"]
        else:
            require("usage" not in packet, "usage on non-terminal chunk")
    require(done, "missing [DONE]")
    return {
        "text": "".join(text), "reasoning": "".join(reasoning), "finish": finish,
        "usage": {key: usage[key] for key in ("prompt_tokens", "completion_tokens", "total_tokens")},
    }


def stream_events(response):
    require(response.headers.get_content_type() == "text/event-stream", "expected SSE response")
    return events(response)


def compare(actual, expected, label):
    require(actual == expected, f"{label}: expected {expected!r}, received {actual!r}")


def check_usage(result, prompt_ids, limit):
    usage = result["usage"]
    compare(usage["prompt_tokens"], len(prompt_ids), "prompt token count")
    count = usage["completion_tokens"]
    require(0 < count <= limit, f"completion usage {count} outside requested limit {limit}")
    compare(usage["total_tokens"], len(prompt_ids) + count, "total token count")


def run_suite(server, args, report):
    def body(**fields):
        return {"model": args.model, "temperature": 0, **fields}

    def chat_body(messages, limit):
        return body(messages=messages, max_tokens=limit, chat_template_kwargs=THINKING_OFF)

    def encode(**source):
        reply = server.json("/v1/token/encode", body(**source))
        compare(reply["length"], len(reply["tokens"]), "tokenizer result length")
        return reply["tokens"]

    def request(label, path, payload, streaming=False):
        case = report["cases"][label] = {"request": {**payload, "stream": streaming}}
        if streaming:
            case["events"] = frames = []
            with server.request(path, {**payload, "stream": True}) as response:
                for frame in stream_events(response):
                    frames.append(frame)
            result = fold_stream(frames, path == CHAT_PATH)
        else:
            case["response"] = server.json(path, payload)
            result = normalize(case["response"], path == CHAT_PATH)
        case["result"] = result
        return result

    report["models"] = server.json("/v1/models")
    model = next((row for row in report["models"]["data"] if row["id"] == args.model), None)
    require(model and any(engine["state"] == "ready" for engine in model["engines"]),
            f"model {args.model} must be ready")
    raw_tokens = encode(text=RAW_PROMPT)
    report["raw_prompt_ids"] = raw_tokens
    compare(raw_tokens, RAW_PROMPT_IDS, "pinned raw prompt IDs")
    raw = body(prompt=RAW_PROMPT, max_tokens=3)
    baseline = request("raw", RAW_PATH, raw)
    compare(baseline["text"], RAW_PREFIX, "reference raw text prefix")
    report["raw_reference_prefix_ids"] = encode(text=baseline["text"])
    compare(report["raw_reference_prefix_ids"], RAW_PREFIX_IDS, "re-encoded reference prefix")
    check_usage(baseline, raw_tokens, 3)
    compare(baseline["finish"], "length", "raw output cap")
    compare(baseline["usage"]["completion_tokens"], 3, "raw cap token count")
    compare(request("raw_consecutive", RAW_PATH, raw), baseline, "consecutive raw result")
    compare(request("raw_stream", RAW_PATH, raw, True), baseline, "raw stream parity")
    one = request("raw_one_token", RAW_PATH, {**raw, "max_tokens": 1})
    check_usage(one, raw_tokens, 1)
    compare((one["text"], one["finish"]), (" Paris", "length"), "one-token cap")

    messages = [{"role": "user", "content": "Name the capital of France. Answer in one word."}]
    chat = chat_body(messages, 64)
    chat_ids = encode(messages=messages, chat_template_kwargs=THINKING_OFF)
    report["chat_prompt_ids"] = chat_ids
    result = request("chat_eos", CHAT_PATH, {**chat, "x-sample-trace": True})
    check_usage(result, chat_ids, 64)
    compare(result["text"], "Paris", "chat answer")
    require(result["reasoning"], "expected separate chat reasoning")
    compare(result["finish"], "stop", "chat EOS finish")
    case = report["cases"]["chat_eos"]
    query = urlencode({"id": case["response"]["id"]})
    trace = case["sampler_trace"] = server.json(f"/v1/sampler-trace?{query}")
    ids = [position["token_id"] for position in trace["positions"]]
    compare(trace["reply_id"], case["response"]["id"], "sampler trace request ID")
    require(not trace["truncated"] and not trace["alignment_error"], "incomplete sampler trace")
    compare(ids.count(EOS_ID), 1, "EOS count")
    compare(ids[-1:], [EOS_ID], "terminal sampled token")
    compare(len(ids), result["usage"]["completion_tokens"], "usage includes the sampled EOS")
    compare(request("chat_stream", CHAT_PATH, chat, True), result, "chat stream parity")

    limited_chat = chat_body(messages, 3)
    limited = request("chat_limit", CHAT_PATH, limited_chat, True)
    check_usage(limited, chat_ids, 3)
    compare(limited["finish"], "length", "chat output cap")
    compare(limited["usage"]["completion_tokens"], 3, "chat cap token count")

    # The full history is supplied explicitly on both requests.
    history = [*messages, {"role": "assistant", "content": result["text"]},
               {"role": "user", "content": "Repeat your previous answer in one word."}]
    history_ids = encode(messages=history, chat_template_kwargs=THINKING_OFF)
    report["multiturn_prompt_ids"] = history_ids
    require(len(history_ids) > len(chat_ids), "history did not grow the encoded prompt")
    followup = chat_body(history, 64)
    second = request("multiturn", CHAT_PATH, followup)
    check_usage(second, history_ids, 64)
    compare(second["text"].strip().rstrip("."), "Paris", "answer using prior turn")
    compare(request("multiturn_stream", CHAT_PATH, followup, True), second, "multiturn stream parity")
    compare(request("raw_after_chat", RAW_PATH, raw), baseline, "reset after longer chat requests")

    cancellation = report["cases"]["cancel"] = {}
    payload = cancellation["request"] = {**raw, "max_tokens": 64, "stream": True}
    with server.request(RAW_PATH, payload) as response:
        for frame in stream_events(response):
            require(frame != "[DONE]", "stream completed before it could be cancelled")
            packet = json.loads(frame)
            require("error" not in packet, f"stream error before cancellation: {packet}")
            choice = choice_of(packet)
            require(choice.get("finish_reason") is None, "stream finished before socket close")
            if choice["text"]:
                cancellation["first_chunk"] = packet
                break
        else:
            raise ValueError("stream closed before it could be cancelled")
    request_id = cancellation["first_chunk"]["id"]
    deadline = time.monotonic() + args.cancel_timeout
    while time.monotonic() < deadline:
        rows = server.json("/v1/recent-requests", timeout=args.cancel_timeout)["requests"]
        row = next((row for row in rows if row["request_id"] == request_id), None)
        if row is not None:
            cancellation["request_log"] = row
            compare(row["outcome"], "cancelled", "socket-close request outcome")
            compare(row["finish_reason"], None, "cancelled request finish reason")
            break
        time.sleep(0.2)
    else:
        raise ValueError(f"no request log for cancelled stream {request_id}")
    compare(request("raw_after_cancel", RAW_PATH, raw), baseline, "reset after cancellation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="server origin, e.g. http://127.0.0.1:18795")
    parser.add_argument("--model", required=True, help="model ID from /v1/models")
    parser.add_argument("--api-key-env", default="ENGINESRV_API_KEY")
    parser.add_argument("--timeout", type=float, default=300, help="HTTP socket timeout in seconds")
    parser.add_argument("--cancel-timeout", type=float, default=30, help="seconds to await cancellation")
    parser.add_argument("--report", required=True, type=Path, help="JSON evidence written on pass or failure")
    args = parser.parse_args()
    url = urlsplit(args.base_url)
    if (url.scheme not in ("http", "https") or not url.hostname or url.username is not None
            or url.path not in ("", "/") or url.query or url.fragment):
        parser.error("--base-url must be an HTTP(S) origin")
    key = os.environ.get(args.api_key_env)
    if not key:
        parser.error(f"set {args.api_key_env} to the server API key")
    if min(args.timeout, args.cancel_timeout) <= 0:
        parser.error("timeouts must be positive")
    report = {"base_url": args.base_url, "model": args.model, "cases": {}}
    # Open the artifact before contacting the server so an invalid path fails early.
    with args.report.open("w", encoding="utf-8") as output:
        try:
            run_suite(Server(url, key, args.timeout), args, report)
        except (ValueError, KeyError, TypeError, OSError, http.client.HTTPException) as error:
            report["error"] = f"{type(error).__name__}: {error}"
        finally:
            json.dump(report, output, indent=2)
            output.write("\n")
    if "error" in report:
        parser.exit(1, f"{report['error']}\nEvidence: {args.report}\n")
    print(f"Passed {len(report['cases'])} Inkling serving cases. Evidence: {args.report}")


if __name__ == "__main__":
    main()
