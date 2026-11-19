#!/usr/bin/env python3
"""Measure two overlapping chat clients on a running FiFi endpoint."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import bench_engines as bench
import bench_long_context as common


@dataclass
class Prompt:
    messages: list
    tokens: int
    budget: int
    fingerprint: str


def prompt(client, model, messages, budget):
    encoded = client.json("/v1/token/encode", {"model": model, "messages": messages,
                                                "x-thinking": "off"})
    ids = encoded["tokens"]
    if not isinstance(ids, list) or len(ids) != encoded["length"] or not ids:
        raise ValueError("invalid tokenized fixture")
    digest = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
    return Prompt(messages, len(ids), budget, digest)


def make_pair(client, model, depth, seed):
    rng = random.Random(seed)
    codes = tuple(str(n) for n in rng.sample(range(100000, 1000000), 3))
    answer = "|".join(codes)
    cold, _, _ = common.size_prompt(client, model, depth - 192, 128, seed, codes)
    short, _, _ = common.size_prompt(client, model, 640, 64, seed + "-count", codes)
    conversation = common.messages(short) + [
        {"role": "assistant", "content": answer},
        {"role": "user", "content": "Count from 1 to 200, one number per line, without skipping numbers."},
    ]
    return (prompt(client, model, common.messages(cold), 64),
            prompt(client, model, conversation, 128), answer)


def timed_stream(client, model, item, notify):
    body = {"model": model, "messages": item.messages, "max_tokens": item.budget,
            "temperature": 0, "stream": True, "x-thinking": "off"}
    fragments, arrivals = [], []
    start = time.monotonic()
    with client.request("/v1/chat/completions", body) as response:
        headers = time.monotonic()
        notify("admitted")

        def capture():
            for raw in response:
                if raw.startswith(b"data: ") and raw[6:].strip() != b"[DONE]":
                    for choice in json.loads(raw[6:]).get("choices", []):
                        delta = choice.get("delta") or {}
                        if delta.get("content"):
                            arrivals.append(time.monotonic())
                            fragments.append(delta["content"])
                            if len(arrivals) == 1:
                                notify("visible")
                yield raw

        stream = bench.parse_stream(capture(), start)
        end = time.monotonic()
    if not arrivals or stream["usage"]["prompt_tokens"] != item.tokens:
        raise ValueError("missing visible output or prompt-token count mismatch")
    return dict(stream=stream, pieces=fragments, times=arrivals, start=start,
                admitted=headers, end=end)


def run_pair(client, model, prompts, first_name):
    second_name = "cold" if first_name == "count" else "count"
    phase = "visible" if first_name == "count" else "admitted"
    future = None
    with ThreadPoolExecutor(max_workers=1) as pool:
        def notify(event):
            nonlocal future
            if event == phase and future is None:
                future = pool.submit(timed_stream, client, model, prompts[second_name], lambda _: None)

        first = timed_stream(client, model, prompts[first_name], notify)
        if future is None:
            raise ValueError("first client did not reach the launch phase")
        second = future.result()
    begin = first["times"][0] if first_name == "count" else first["admitted"]
    cutoff = first["times"][-1] if first_name == "count" else first["times"][0]
    if not begin <= second["start"] <= second["admitted"] < cutoff:
        raise ValueError("second request was not admitted during the first decode or cold prefill")
    return {first_name: first, second_name: second}


def checked_reuse(usage):
    direct = usage.get("reused_tokens")
    details = usage.get("prompt_tokens_details", {}).get("cached_tokens")
    reused = bench.reused_tokens(usage)
    if reused is None or any(v is not None and (not bench.nonnegative_int(v) or v != reused)
                             for v in (direct, details)):
        raise ValueError("missing or inconsistent cached-token counts")
    return reused


def p95(values):
    return sorted(values)[math.ceil(0.95 * len(values)) - 1] if len(values) >= 20 else None


def summary(client, model, result, item, reference):
    stream, times = result["stream"], result["times"]
    usage = stream["usage"]
    metrics = common.recent_server_metrics(client, model, stream)
    if metrics["server_queue_wait_ms"] is None:
        raise ValueError("no matching server queue measurement for " + stream["request_id"])
    offsets = lambda t: round((t - reference) * 1000, 3)
    gaps = [(b - a) * 1000 for a, b in zip(times, times[1:])]
    return {"id": stream["request_id"], "prompt_sha256": item.fingerprint,
            "prompt_tokens": item.tokens, "reused_tokens": checked_reuse(usage),
            "completion_tokens": usage["completion_tokens"], "finish": stream["finish_reason"],
            "reply_sha256": hashlib.sha256("".join(result["pieces"]).encode()).hexdigest(),
            "submit_ms": offsets(result["start"]), "admitted_ms": offsets(result["admitted"]),
            "first_visible_ms": offsets(times[0]), "last_visible_ms": offsets(times[-1]),
            "done_ms": offsets(result["end"]), "ttft_ms": (times[0] - result["start"]) * 1000,
            "visible_gaps": len(gaps), "p95_visible_gap_ms": p95(gaps), **metrics}


def valid_count(result):
    text = "".join(result["pieces"])
    expected = "\n".join(map(str, range(1, 201)))
    return (len(text.splitlines()) >= 20 and expected.startswith(text)
            and result["stream"]["usage"]["completion_tokens"] == 128
            and result["stream"]["finish_reason"] == "length")


def measure(client, model, depth, seed, first):
    cold, count, answer = make_pair(client, model, depth, seed)
    if max(cold.tokens + cold.budget, count.tokens + count.budget) > depth:
        raise ValueError("fixtures exceed the requested context budget")
    control = timed_stream(client, model, count, lambda _: None)
    if not valid_count(control):
        raise ValueError("the serial counting control did not produce the expected sequence")
    prompts = {"cold": cold, "count": count}
    results = run_pair(client, model, prompts, first)
    recall, counting = results["cold"], results["count"]
    if ("".join(recall["pieces"]).strip() != answer or recall["stream"]["finish_reason"] != "stop"
            or not valid_count(counting) or "".join(counting["pieces"]) != "".join(control["pieces"])):
        raise ValueError("concurrent output failed the recall or serial counting control")
    origin = results[first]["start"]
    clients = {name: summary(client, model, result, prompts[name], origin)
               for name, result in results.items()}
    if clients["cold"]["reused_tokens"] or (first == "count" and not clients["count"]["reused_tokens"]):
        raise ValueError("measured cache state did not match the workload")
    duration = max(row["end"] for row in results.values()) - origin
    return {"context": depth, "first": first, "seed": seed, "clients": clients,
            "serial_count_control": summary(client, model, control, count, control["start"]),
            "pair_seconds": duration,
            "aggregate_output_tokens_per_second": sum(r["completion_tokens"] for r in clients.values()) / duration}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--contexts", type=int, nargs="+", required=True)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--api-key-env")
    parser.add_argument("--output", type=Path, required=True, help="new JSONL file, one row per pair")
    args = parser.parse_args()
    if args.repetitions < 1 or min(args.contexts) < 2048 or len(set(args.contexts)) != len(args.contexts):
        parser.error("use positive repetitions and distinct context budgets of at least 2048")
    client = common.Client(args.url, os.environ[args.api_key_env] if args.api_key_env else None, args.timeout)
    limit = common.context_limit(client, args.model)
    with args.output.open("x", buffering=1) as out:
        for depth in args.contexts:
            if depth > limit:
                out.write(json.dumps({"model": args.model, "context": depth, "server_limit": limit,
                                      "state": "above_advertised_limit"}) + "\n")
                continue
            for i in range(args.repetitions):
                for first in (("count", "cold") if i % 2 == 0 else ("cold", "count")):
                    seed = f"{args.seed}/{depth}/{i}/{first}"
                    row = measure(client, args.model, depth, seed, first)
                    out.write(json.dumps({"model": args.model, **row}, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
