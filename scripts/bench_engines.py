#!/usr/bin/env python3
"""Measure prefill throughput, TTFT and decode throughput of a running enginesrv.

Usage: bench_engines.py BASE_URL API_KEY MODEL [PROMPT_TOKENS ...]

For each prompt size the script makes two requests: a max_tokens=1 completion
(wall time ~= prefill; usage gives the exact prompt token count) and a
streaming completion whose first content delta gives TTFT and whose remaining
deltas give decode tok/s. The streamed request re-prefills the same prompt,
so TTFT benefits from any warm-prefix behavior the engine has; treat the
max_tokens=1 wall time as the cold-prefill number.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request

PARAGRAPH = (
    "The mountain weather shifted through the afternoon: thin cloud over the ridge, "
    "a cold wind funneling up the valley, and by evening the first spindrift off the summit. "
)


def bench(base: str, key: str, model: str, approx_tokens: int) -> None:
    prompt = PARAGRAPH * (approx_tokens // 25 + 1)
    headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "x-thinking": "off"}

    t0 = time.monotonic()
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 json.dumps({**body, "max_tokens": 1}).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=600) as r:
        usage = json.load(r).get("usage", {})
    prefill_s = time.monotonic() - t0
    prompt_tokens = usage.get("prompt_tokens", approx_tokens)

    t0 = time.monotonic()
    first = None
    tokens = 0
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 json.dumps({**body, "max_tokens": 96, "stream": True}).encode(),
                                 headers=headers)
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            for choice in json.loads(line[6:]).get("choices", []):
                if choice.get("delta", {}).get("content"):
                    tokens += 1
                    if first is None:
                        first = time.monotonic()
    ttft = (first or time.monotonic()) - t0
    decode = tokens / (time.monotonic() - first) if first and tokens > 8 else float("nan")
    print(f"| {prompt_tokens:>7} | {prompt_tokens / prefill_s:>8.0f} | {prefill_s:>6.1f}s | {ttft:>6.2f}s | {decode:>6.1f} |")


def main() -> None:
    if len(sys.argv) < 5:
        sys.exit(__doc__)
    base, key, model = sys.argv[1:4]
    print("| prompt_t | prefill t/s | prefill | TTFT | decode t/s |")
    print("|---:|---:|---:|---:|---:|")
    for size in [int(x) for x in sys.argv[4:]]:
        bench(base, key, model, size)


if __name__ == "__main__":
    main()
