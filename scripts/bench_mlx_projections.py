#!/usr/bin/env python3
"""Compare same-shape reshapes feeding MLX packed projections."""

import argparse
import csv
import json
import os
import re
import subprocess
from pathlib import Path

ENGINE = Path(__file__).resolve().parents[1] / "engine"
TEST = "packed_projection_matrix"
RECORD = "mlx_projection "
FORMATS = [("affine", bits, group) for bits in (2, 3, 4, 5, 6, 8) for group in (32, 64, 128)]
FORMATS += [("mxfp4", 4, 32), ("mxfp8", 8, 32)]
CASES = [(mode, bits, group, dtype, rows)
         for mode, bits, group in FORMATS for dtype in ("f32", "bf16") for rows in (1, 32)]


def parse(log, samples):
    rows = [json.loads(line.split(RECORD, 1)[1]) for line in log.splitlines() if RECORD in line]
    keys = [tuple(row[name] for name in ("quant", "bits", "group", "dtype", "rows")) for row in rows]
    if keys != CASES:
        raise RuntimeError("missing or out-of-order packed projection cases")
    for row in rows:
        if row["width"] != 2048 or not re.fullmatch("[0-9a-f]{64}", row["output_sha256"]):
            raise RuntimeError(f"bad packed projection case {row}")
        submit, elapsed = row["submit_ns"], row["elapsed_ns"]
        if len(submit) != samples or len(elapsed) != samples or any(
            not 0 < submit_ns < elapsed_ns for submit_ns, elapsed_ns in zip(submit, elapsed)
        ):
            raise RuntimeError(f"bad time samples {row}")
    return rows


def collect(binary, samples, smoke):
    listed = subprocess.run([binary, "--ignored", "--list"], cwd=ENGINE,
                            check=True, capture_output=True, text=True).stdout.splitlines()
    if listed.count(f"{TEST}: test") != 1:
        raise RuntimeError(f"expected one ignored test named {TEST}")
    expected = {}
    results = []
    for run, mode in enumerate((0, 1) if smoke else (0, 1, 1, 0), 1):
        env = {**os.environ, "FIFI_MLX_REUSE_IDENTITY_RESHAPE": str(mode),
               "FIFI_MLX_PROJECTION_SAMPLES": str(samples)}
        completed = subprocess.run(
            [binary, "--ignored", "--exact", "--nocapture", "--test-threads=1", TEST],
            cwd=ENGINE, env=env, capture_output=True, text=True, timeout=120,
        )
        if completed.returncode or "test result: ok. 1 passed;" not in completed.stdout:
            raise RuntimeError(completed.stdout[-4000:] + completed.stderr[-4000:])
        for key, row in zip(CASES, parse(completed.stdout, samples)):
            digest = row["output_sha256"]
            if digest != expected.setdefault(key, digest):
                raise RuntimeError(f"packed projection output changed at {key}")
            results.append({
                "run": run, "reuse_identity_reshape": mode, **row,
                "submit_ns": json.dumps(row["submit_ns"], separators=(",", ":")),
                "elapsed_ns": json.dumps(row["elapsed_ns"], separators=(",", ":")),
            })
        print(f"Run {run}: all {len(CASES)} packed projection cases matched", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--samples", type=int, default=12)
    args = parser.parse_args()
    if args.samples < 1 or args.output.exists():
        parser.error("use a positive sample count and a new output path")
    rows = collect(args.binary.resolve(), args.samples, args.smoke)
    with args.output.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
