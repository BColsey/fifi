#!/usr/bin/env python3
"""Measure two identical greedy Inkling requests in a fresh server process."""

import argparse
import ctypes
import hashlib
import http.client
import json
import os
from pathlib import Path
import platform
import re
import secrets
import subprocess
import sys
import time
from urllib.parse import urlsplit

from check_inkling_serving import RAW_PATH, RAW_PROMPT, Server, require

MODEL = "inkling"
KEY_ENV = "FIFI_INKLING_BENCH_KEY"
LISTENING = re.compile(r"^enginesrv listening on (127\.0\.0\.1:[1-9][0-9]*)$", re.MULTILINE)
SWAP_FIELDS = ("Swapins", "Swapouts")
RECEIPT_ENV = (
    "FIFI_INKLING2_MINI_MLX", "FIFI_INKLING2_MINI_GGUF_MLX",
    "FIFI_INKLING_GGUF_PACKED_MLX", "FIFI_INKLING_GGUF_MMAP_DENSE",
    "FIFI_INKLING_ATTN_CPU_WORKERS", "FIFI_INKLING_PREFILL_ROWS",
    "FIFI_INKLING_MMAP_DENSE", "FIFI_INKLING_FILE_CACHE_ONLY",
    "FIFI_INKLING_ROUTE_TRACE", "FIFI_INKLING_EXPERT_TIMING",
    "FIFI_INKLING_COMPONENT_TIMING", "FIFI_MLX_AUTO_BUDGET",
    "FIFI_MLX_WORKING_SET_MIB",
)


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def swap_pages():
    if sys.platform != "darwin":
        return None
    raw = subprocess.check_output(["/usr/bin/vm_stat"], text=True)
    values = {}
    for line in raw.splitlines():
        name, _, count = line.partition(":")
        if name in SWAP_FIELDS:
            values[name] = int(count.strip().rstrip("."))
    require(set(values) == set(SWAP_FIELDS), "vm_stat did not report both swap counters")
    return values


def disk_reader():
    if sys.platform != "darwin":
        return lambda pid: None

    class ResourceUsageV2(ctypes.Structure):
        _fields_ = [
            ("uuid", ctypes.c_uint8 * 16),
            ("other_counters", ctypes.c_uint64 * 16),
            ("disk_bytes_read", ctypes.c_uint64),
            ("disk_bytes_written", ctypes.c_uint64),
        ]

    libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    get_usage = libproc.proc_pid_rusage
    get_usage.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.POINTER(ResourceUsageV2))
    get_usage.restype = ctypes.c_int

    def measure(pid):
        usage = ResourceUsageV2()
        if get_usage(pid, 2, ctypes.byref(usage)) != 0:  # RUSAGE_INFO_V2
            error = ctypes.get_errno()
            raise OSError(error, f"proc_pid_rusage({pid}): {os.strerror(error)}")
        return usage.disk_bytes_read

    return measure


def environment(args):
    env = os.environ.copy()
    env.setdefault("FIFI_INKLING2_MINI_MLX", "1")
    env.setdefault("FIFI_INKLING_EXPERT_TIMING", "1")
    if args.ceiling_mib == "auto":
        env.pop("FIFI_MLX_WORKING_SET_MIB", None)
        env["FIFI_MLX_AUTO_BUDGET"] = "1"
    elif args.ceiling_mib is not None:
        env["FIFI_MLX_WORKING_SET_MIB"] = args.ceiling_mib
    env[KEY_ENV] = secrets.token_urlsafe(32)
    return env


def wait_until_ready(process, logfile, key, args, report, started):
    deadline = started + args.startup_timeout
    server = None
    while time.monotonic() < deadline:
        require(process.poll() is None, f"server exited before readiness with status {process.returncode}")
        if server is None:
            match = LISTENING.search(logfile.read_text(encoding="utf-8", errors="replace"))
            if match:
                report["listen"] = match[1]
                server = Server(urlsplit(f"http://{match[1]}"), key, args.request_timeout)
        if server is not None:
            try:
                system = server.json("/v1/system", timeout=min(2, max(0.01, deadline - time.monotonic())))
            except (OSError, http.client.HTTPException):
                time.sleep(0.1)
                continue
            report["system"] = system
            engines = system["engines"]
            require(len(engines) == 1, f"expected one engine, got {engines}")
            engine = engines[0]
            require(engine["id"] == MODEL and engine["model"] == MODEL,
                    f"unexpected engine identity {engine}")
            require(engine["context_limit"] == args.context, f"unexpected context {engine}")
            if engine["state"] == "ready":
                require("mlx" in system["backends_compiled"], "ready server lacks compiled MLX backend")
                report["ready_seconds"] = time.monotonic() - started
                return server
            require(engine["state"] == "loading", f"unexpected engine state {engine}")
        time.sleep(0.1)
    raise TimeoutError(f"engine did not become ready within {args.startup_timeout:g}s")


def validate_sample(sample, prompt_count, limit):
    response, rates = sample["response"], sample["rates"]
    usage = response["usage"]
    require(response["model"] == MODEL and response["engine"] == MODEL, "unexpected reply model")
    choices = response["choices"]
    require(response["object"] == "text_completion" and len(choices) == 1
            and choices[0]["index"] == 0, "expected one raw choice")
    choice = choices[0]
    require(isinstance(choice["text"], str) and choice["text"], "reply has no text")
    require(choice["finish_reason"] == "length", f"expected output cap, got {choice['finish_reason']}")
    require(usage["prompt_tokens"] == prompt_count and usage["completion_tokens"] == limit
            and usage["total_tokens"] == prompt_count + limit, f"unexpected reply usage {usage}")
    require(rates["request_id"] == response["id"] and rates["outcome"] == "ok"
            and rates["prompt_tokens"] == prompt_count and rates["completion_tokens"] == limit
            and rates["finish_reason"] == choice["finish_reason"], f"request log disagrees with reply: {rates}")


def run_requests(server, pid, request, context, report):
    read_disk = disk_reader()
    encoded = server.json("/v1/token/encode", {"model": MODEL, "text": request["prompt"]})
    ids = report["prompt_ids"] = encoded["tokens"]
    limit = request["max_tokens"]
    require(isinstance(ids, list) and ids and all(type(token) is int and token >= 0 for token in ids)
            and encoded["length"] == len(ids), "invalid encoded prompt")
    require(len(ids) + limit <= context, "prompt and requested output exceed the context")
    for order in ("first", "repeat"):
        sample = {"order": order}
        report["samples"].append(sample)
        swap_start = swap_pages()
        disk_start = read_disk(pid)
        started = time.perf_counter()
        response = sample["response"] = server.json(RAW_PATH, request)
        sample["wall_seconds"] = time.perf_counter() - started
        disk_end = read_disk(pid)
        swap_end = swap_pages()
        sample["swap_page_delta"] = (
            {name: swap_end[name] - swap_start[name] for name in SWAP_FIELDS}
            if swap_end is not None else None
        )
        require(disk_start is None or disk_end >= disk_start, "process disk byte counter decreased")
        sample["attributed_disk_read_bytes"] = disk_end - disk_start if disk_end is not None else None
        recent = server.json("/v1/recent-requests")["requests"]
        sample["rates"] = next((r for r in recent if r["request_id"] == response["id"]), None)
        require(sample["rates"] is not None, f"missing recent-request row for {response['id']}")
        validate_sample(sample, len(ids), limit)
    first, repeat = (sample["response"] for sample in report["samples"])
    require(first["id"] != repeat["id"], "first and repeat share a request ID")
    require(first["choices"] == repeat["choices"], "greedy first and repeat outputs differ")


def stop(process):
    if process.poll() is not None:
        return {"exit_code": process.returncode, "forced": False, "already_exited": True}
    process.terminate()
    try:
        code = process.wait(timeout=10)
        return {"exit_code": code, "forced": False}
    except subprocess.TimeoutExpired:
        process.kill()
        return {"exit_code": process.wait(), "forced": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True, help="path to a built enginesrv binary")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True, help="new directory for config, log and receipt")
    parser.add_argument("--prompt", default=RAW_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=12)
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--ceiling-mib", help="override FIFI_MLX_WORKING_SET_MIB, or use auto")
    parser.add_argument("--startup-timeout", type=float, default=120)
    parser.add_argument("--request-timeout", type=float, default=300)
    parser.add_argument("--commit", help="source revision for this binary")
    parser.add_argument("--build", help="build flags or build receipt identifier")
    parser.add_argument("--checkpoint-id", help="artifact revision or precomputed digest")
    args = parser.parse_args()
    if not args.prompt or not (1 <= args.max_tokens < args.context) or args.context > 1_048_576:
        parser.error("provide a prompt and an output limit smaller than the supported context")
    if min(args.startup_timeout, args.request_timeout) <= 0:
        parser.error("timeouts must be positive")
    if args.ceiling_mib not in (None, "auto") and not (
        args.ceiling_mib.isascii() and args.ceiling_mib.isdecimal()
        and 0 < int(args.ceiling_mib) < 1 << 44
    ):
        parser.error("--ceiling-mib must be a positive whole MiB value or auto")

    binary = args.server.resolve()
    model = args.model.absolute()
    tokenizer = (args.tokenizer_dir or (model.parent if model.is_file() else model)).absolute()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        parser.error(f"server is not executable: {binary}")
    if not model.is_file() and not model.is_dir():
        parser.error(f"model does not exist: {model}")
    if not tokenizer.is_dir():
        parser.error(f"tokenizer directory does not exist: {tokenizer}")
    try:
        args.output_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
    except OSError as error:
        parser.error(f"cannot create benchmark output directory: {error}")
    logfile = args.output_dir / "server.log"
    configfile = args.output_dir / "server.json"
    receipt = args.output_dir / "receipt.json"
    env = environment(args)
    config = {
        "listen": "127.0.0.1:0", "api_key_env": KEY_ENV, "prefix_reuse": False,
        "engines": [{"id": MODEL, "family": "inkling2_mini", "model": MODEL,
                     "model_dir": str(model), "tokenizer_dir": str(tokenizer),
                     "context_limit": args.context}],
    }
    configfile.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    report = {
        "status": "failed", "machine": platform.machine(), "platform": platform.platform(),
        "binary": str(binary), "binary_sha256": digest_file(binary),
        "commit": args.commit, "build": args.build, "checkpoint_id": args.checkpoint_id,
        "checkpoint_file_bytes": model.stat().st_size if model.is_file() else None,
        "config": config, "server_log": logfile.name,
        "environment": {k: env[k] for k in RECEIPT_ENV if k in env},
        "os_cache": "uncontrolled, no purge",
        "swap_page_bytes": os.sysconf("SC_PAGE_SIZE") if sys.platform == "darwin" else None,
        "request": {"model": MODEL, "prompt": args.prompt, "max_tokens": args.max_tokens, "temperature": 0},
        "samples": [],
    }
    try:
        with logfile.open("w", encoding="utf-8") as output:
            started = time.monotonic()
            process = subprocess.Popen(
                [str(binary), "--config", str(configfile.resolve())],
                stdout=output, stderr=subprocess.STDOUT, env=env,
            )
            try:
                server = wait_until_ready(process, logfile, env[KEY_ENV], args, report, started)
                run_requests(server, process.pid, report["request"], args.context, report)
            finally:
                report["shutdown"] = stop(process)
        require(report["shutdown"] == {"exit_code": 0, "forced": False}, "server did not shut down cleanly")
        report["status"] = "passed"
    except (ValueError, KeyError, TypeError, OSError, subprocess.SubprocessError, http.client.HTTPException) as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        receipt.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if report["status"] != "passed":
        parser.exit(1, f"{report['error']}\nEvidence: {receipt}\n")
    print(f"Measured two matching requests. Evidence: {receipt}")


if __name__ == "__main__":
    main()
