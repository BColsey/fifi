#!/usr/bin/env python3
"""Measure a running enginesrv without treating streaming chunks as tokens.

Usage: bench_engines.py BASE_URL API_KEY MODEL PROMPT_TOKENS [PROMPT_TOKENS ...]

For each size, the first request generates one token. Prompt tokens divided by
its full wall time is an estimate that also includes tokenization, queueing,
reply work and any prefix reuse; it is not necessarily a cold-prefill rate.
The second request streams a reply. Its first visible content/reasoning delta
measures client-observed latency, not necessarily the first generated token.
Decode throughput comes only from a matching server recent-request record.
If that record is unavailable or cannot be matched, decode is shown as n/a.
"""

from __future__ import annotations

import json
import math
import sys
import time
import urllib.request

PARAGRAPH = (
    "The mountain weather shifted through the afternoon: thin cloud over the ridge, "
    "a cold wind funneling up the valley, and by evening the first spindrift off the summit. "
)


def nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def parse_stream(lines, start: float, clock=time.monotonic) -> dict:
    """Read a complete enginesrv SSE reply; never infer token counts from chunks."""
    request_id = None
    first_delta = None
    usage = None
    finish = None
    done = False
    event = None
    for raw in lines:
        line = raw.decode("utf-8").strip()
        if line.startswith("event: "):
            event = line[7:]
            continue
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            done = True
            break
        obj = json.loads(data)
        if event == "error":
            raise RuntimeError("the server returned a streaming error")
        event = None
        incoming_id = obj.get("id")
        if incoming_id:
            if request_id is not None and request_id != incoming_id:
                raise ValueError("the response ID changed during the stream")
            request_id = incoming_id
        for choice in obj.get("choices") or []:
            delta = choice.get("delta") or {}
            if first_delta is None and (
                delta.get("content") or delta.get("reasoning_content")
            ):
                first_delta = clock() - start
            if choice.get("finish_reason") is not None:
                finish = choice["finish_reason"]
        if obj.get("usage") is not None:
            usage = obj["usage"]
    if (
        not done
        or not isinstance(request_id, str)
        or not request_id
        or not isinstance(finish, str)
        or not finish
        or not isinstance(usage, dict)
        or not nonnegative_int(usage.get("prompt_tokens"))
        or not nonnegative_int(usage.get("completion_tokens"))
    ):
        raise ValueError(
            "stream ended without a complete ID, finish reason and token usage"
        )
    return {
        "request_id": request_id,
        "first_delta_seconds": first_delta,
        "finish_reason": finish,
        "usage": usage,
    }


def matching_server_rate(rows, model: str, stream: dict) -> float | None:
    """Accept only a unique server row for this model and completed response."""
    if not isinstance(rows, list):
        return None
    matches = [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("request_id") == stream["request_id"]
        and row.get("model") == model
    ]
    if len(matches) != 1:
        return None
    row = matches[0]
    usage = stream["usage"]
    rate = row.get("decode_tokens_per_second")
    if (
        row.get("status") != 200
        or row.get("outcome") != "ok"
        or row.get("stream") is not True
        or row.get("finish_reason") != stream["finish_reason"]
        or not nonnegative_int(row.get("prompt_tokens"))
        or not nonnegative_int(row.get("completion_tokens"))
        or row.get("prompt_tokens") != usage["prompt_tokens"]
        or row.get("completion_tokens") != usage["completion_tokens"]
        or isinstance(rate, bool)
        or not isinstance(rate, (int, float))
        or not math.isfinite(rate)
        or rate <= 0
    ):
        return None
    return float(rate)


def recent_server_rate(
    base: str, headers: dict, model: str, stream: dict
) -> float | None:
    """The request log may arrive just after the final SSE event."""
    deadline = time.monotonic() + 5
    while True:
        req = urllib.request.Request(base + "/v1/recent-requests", headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.load(response)
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict):
            return None
        rate = matching_server_rate(data.get("requests"), model, stream)
        if rate is not None or time.monotonic() >= deadline:
            return rate
        time.sleep(0.1)


def reused_tokens(usage: dict) -> int | None:
    cached = usage.get("reused_tokens")
    if not nonnegative_int(cached):
        details = usage.get("prompt_tokens_details")
        cached = details.get("cached_tokens") if isinstance(details, dict) else None
    if not nonnegative_int(cached):
        return None
    if cached > usage["prompt_tokens"]:
        raise ValueError("reused token count exceeds prompt token count")
    return cached


def bench(base: str, key: str, model: str, approx_tokens: int) -> dict:
    if approx_tokens <= 0:
        raise ValueError("prompt size must be positive")
    base = base.rstrip("/")
    prompt = PARAGRAPH * (approx_tokens // 25 + 1)
    headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "x-thinking": "off",
    }

    started = time.monotonic()
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        json.dumps({**body, "max_tokens": 1}).encode(),
        headers=headers,
    )
    with urllib.request.urlopen(req, timeout=600) as response:
        usage = json.load(response).get("usage")
    prefill_s = time.monotonic() - started
    if (
        not isinstance(usage, dict)
        or not nonnegative_int(usage.get("prompt_tokens"))
        or usage["prompt_tokens"] == 0
        or not nonnegative_int(usage.get("completion_tokens"))
        or usage["completion_tokens"] != 1
        or prefill_s <= 0
    ):
        raise ValueError("the first request did not report valid prompt token usage")
    cached = reused_tokens(usage)

    started = time.monotonic()
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        json.dumps({**body, "max_tokens": 96, "stream": True}).encode(),
        headers=headers,
    )
    with urllib.request.urlopen(req, timeout=600) as response:
        stream = parse_stream(response, started)
    if stream["usage"]["prompt_tokens"] != usage["prompt_tokens"]:
        raise ValueError("prompt token count changed between the two requests")
    rate = recent_server_rate(base, headers, model, stream)
    return {
        "prompt_tokens": usage["prompt_tokens"],
        "estimated_prefill_tps": usage["prompt_tokens"] / prefill_s,
        "first_request_seconds": prefill_s,
        "first_request_reused_tokens": cached,
        "first_visible_seconds": stream["first_delta_seconds"],
        "stream_reused_tokens": reused_tokens(stream["usage"]),
        "completion_tokens": stream["usage"]["completion_tokens"],
        "finish_reason": stream["finish_reason"],
        "server_decode_tps": rate,
    }


def main() -> None:
    if len(sys.argv) < 5:
        sys.exit(__doc__)
    base, key, model = sys.argv[1:4]
    print(
        "| prompt tokens | estimated prefill t/s | first request | first reuse | first visible | stream reuse | completion tokens | finish | server decode t/s |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|---|---:|")
    for size in [int(x) for x in sys.argv[4:]]:
        result = bench(base, key, model, size)
        reused = (
            str(result["first_request_reused_tokens"])
            if result["first_request_reused_tokens"] is not None
            else "n/a"
        )
        stream_reused = (
            str(result["stream_reused_tokens"])
            if result["stream_reused_tokens"] is not None
            else "n/a"
        )
        first = (
            f"{result['first_visible_seconds']:.2f}s"
            if result["first_visible_seconds"] is not None
            else "n/a"
        )
        decode = (
            f"{result['server_decode_tps']:.2f}"
            if result["server_decode_tps"] is not None
            else "n/a"
        )
        print(
            f"| {result['prompt_tokens']} | {result['estimated_prefill_tps']:.2f} | "
            f"{result['first_request_seconds']:.2f}s | {reused} | {first} | {stream_reused} | "
            f"{result['completion_tokens']} | {result['finish_reason']} | {decode} |"
        )


if __name__ == "__main__":
    main()
