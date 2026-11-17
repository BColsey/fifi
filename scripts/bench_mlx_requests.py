#!/usr/bin/env python3
"""Run paired Flash-Next and Qwen3.6 MLX worker fixtures in fresh processes."""

import argparse
import csv
import json
import os
from pathlib import Path
import re
import subprocess

ENGINE = Path(__file__).resolve().parents[1] / "engine"
CASES = (
    ("flash-next", "flash_next", "FIFI_MLX_FLASH_CHUNKED", ("step", "chunked")),
    ("qwen36", "qwen36", "FIFI_MLX_GROUPED_IMPORT", ("default", "grouped")),
)
PROMPTS = (8, 33)
COMPLETION = 8
RECORD = "mlx_request "


def samples_from(log, family, arm, samples):
    rows = [json.loads(line.split(RECORD, 1)[1]) for line in log.splitlines() if RECORD in line]
    expected = [(prompt, index) for prompt in PROMPTS for index in range(samples)]
    if [(row["prompt_tokens"], row["sample"]) for row in rows] != expected:
        raise RuntimeError(f"{family} {arm}: missing or out-of-order records")
    for row in rows:
        if (
            row["family"] != family
            or row["arm"] != arm
            or row["completion_tokens"] != COMPLETION
            or len(row["tokens"]) != COMPLETION
            or any(
                row[key] <= 0
                for key in ("ttft_ns", "start_to_first_ns", "decode_ns", "metal_syncs",
                            "import_calls", "mlx_peak_active_bytes")
            )
            or row["start_to_first_ns"] > row["ttft_ns"]
        ):
            raise RuntimeError(f"{family} {arm}: invalid record {row}")
    return rows


def collect(binary, pairs, samples):
    listed = subprocess.run(
        [binary, "--ignored", "--list"], cwd=ENGINE, check=True, capture_output=True, text=True
    ).stdout.splitlines()
    records = []
    references = {}
    for family, module, flag, arms in CASES:
        test = f"family::{module}::resident::executor::tests::mlx_request_bench"
        if listed.count(f"{test}: test") != 1:
            raise RuntimeError(f"expected exactly one ignored test named {test}")
        for pair in range(pairs):
            order = (0, 1) if pair % 2 == 0 else (1, 0)
            for position, mode in enumerate(order, start=1):
                arm = arms[mode]
                env = os.environ.copy()
                env.pop("Q36_NO_EMBEDDED", None)
                env.update(
                    Q36_FUSED_PREFILL="1",
                    FIFI_MLX_REQUEST_SAMPLES=str(samples),
                    **{flag: str(mode)},
                )
                run = subprocess.run(
                    ["/usr/bin/time", "-l", binary, "--ignored", "--exact", "--nocapture",
                     "--test-threads=1", test],
                    cwd=ENGINE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, timeout=120,
                )
                if run.returncode or not re.search(r"test result: ok\. 1 passed; 0 failed", run.stdout):
                    raise RuntimeError(run.stdout)
                peak = re.search(r"^\s*(\d+)\s+peak memory footprint$", run.stdout, re.MULTILINE)
                if peak is None or int(peak[1]) == 0:
                    raise RuntimeError(f"{family} {arm}: missing process peak footprint")
                for row in samples_from(run.stdout, family, arm, samples):
                    key = (family, row["prompt_tokens"])
                    if row["tokens"] != references.setdefault(key, row["tokens"]):
                        raise RuntimeError(f"{family} {arm}, pair {pair + 1}: token chain differs")
                    row["tokens"] = json.dumps(row["tokens"], separators=(",", ":"))
                    records.append({
                        "pair": pair + 1, "order": position, **row,
                        "process_peak_footprint_bytes": int(peak[1]),
                    })
            print(f"{family}: pair {pair + 1}/{pairs} passed token parity", flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path, help="release library test executable")
    parser.add_argument("output", type=Path, help="new CSV file for validated samples")
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--samples", type=int, default=8, help="timed requests per prompt and process")
    args = parser.parse_args()
    if args.pairs < 1 or args.samples < 1:
        parser.error("pairs and samples must be positive")
    if args.output.exists():
        parser.error("output already exists")
    records = collect(args.binary.resolve(), args.pairs, args.samples)
    with args.output.open("x", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    print(f"{len(records)} validated requests in {args.output}")


if __name__ == "__main__":
    main()
