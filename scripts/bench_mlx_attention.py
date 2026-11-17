#!/usr/bin/env python3
"""Collect paired Apple Silicon attention timings and allocator peaks."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import subprocess

ENGINE = Path(__file__).resolve().parents[1] / "engine"
DECODE_TEST = "q8_decode_dispatch_timing"
PREFILL_TEST = (
    "family::qwen36::resident::prefill::attn::tests::dense_rep6_flash_pad_bench"
)
DEPTHS = (10240, 11264, 12287, 12288, 12289, 13312, 14336, 15360, 16383, 16384, 16385, 17408)
PREFILL_CASES = (
    (128, 8192, 0), (256, 8192, 0), (512, 8192, 0),
    (128, 16384, 0), (256, 16384, 0), (512, 16384, 0), (512, 16384, 256),
    (256, 32768, 0),
)
MAX_FOOTPRINT = 4 << 30


def enumerate_test(binary, test):
    listing = subprocess.run(
        [binary, "--list", "--ignored", "--exact", test], cwd=ENGINE,
        check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    if listing.count(f"{test}: test") != 1:
        raise RuntimeError(f"expected one ignored test named {test}")


def read_records(log, marker):
    records = [
        json.loads(line.split(marker, 1)[1])
        for line in log.splitlines() if marker in line
    ]
    footprint = re.search(r"^\s*(\d+)\s+peak memory footprint$", log, re.MULTILINE)
    if footprint is None or not 0 < int(footprint[1]) <= MAX_FOOTPRINT:
        raise RuntimeError("missing process footprint or above the 4 GiB run budget")
    return records, int(footprint[1])


def run(binary, test, log_path, marker, settings, first, pairs):
    env = dict(
        os.environ, **settings, FIFI_MLX_ATTN_FLASH_FIRST=str(int(first)),
        FIFI_MLX_ATTN_PAIRS=str(pairs),
    )
    with log_path.open("x") as log:
        result = subprocess.run(
            ["/usr/bin/time", "-l", binary, "--ignored", "--exact",
             "--nocapture", "--test-threads=1", test],
            stdout=log, stderr=subprocess.STDOUT, cwd=ENGINE, env=env, timeout=180,
        )
    log = log_path.read_text()
    if result.returncode or "test result: ok. 1 passed; 0 failed" not in log:
        raise RuntimeError(f"{log_path}\n{log[-3500:]}")
    return read_records(log, marker)


def validate(records, expected, shape_keys, first_key, first, numeric_keys):
    observed = [tuple(row[key] for key in shape_keys) + (row["pair"],) for row in records]
    if observed != expected:
        raise RuntimeError("timing records have missing, duplicate or out-of-order shapes")
    for row in records:
        if row[first_key] != ((row["pair"] % 2 == 0) == first):
            raise RuntimeError(f"incorrect first arm in {row}")
        if any(
            not isinstance(row[key], (float, int)) or not math.isfinite(row[key]) or row[key] < 0
            for key in numeric_keys
        ):
            raise RuntimeError(f"invalid time, memory or error metric in {row}")
        if any(row[key] == 0 for key in numeric_keys if key.endswith(("_ms", "_us"))):
            raise RuntimeError(f"zero call time in {row}")


def write_csv(path, rows):
    with path.open("x", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("decode_binary", type=Path, help="release native_q8_decode test executable")
    parser.add_argument("lib_binary", type=Path, help="release library test executable")
    parser.add_argument("output", type=Path, help="new directory for logs and CSV files")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    decode_binary, lib_binary = args.decode_binary.resolve(), args.lib_binary.resolve()
    output = args.output.resolve()
    enumerate_test(decode_binary, DECODE_TEST)
    enumerate_test(lib_binary, PREFILL_TEST)
    output.mkdir()
    decode_rows, prefill_rows = [], []
    depths = (12288, 16384) if args.smoke else DEPTHS
    cases = (PREFILL_CASES[0], PREFILL_CASES[5], PREFILL_CASES[-1]) if args.smoke else PREFILL_CASES
    n_decode = 1 if args.smoke else 20
    n_prefill = 1 if args.smoke else 6
    for repetition in range(1, (1 if args.smoke else 4) + 1):
        first = repetition % 2 == 1
        depth_order = depths if first else depths[::-1]
        records, footprint = run(
            decode_binary, DECODE_TEST, output / f"decode-{repetition}.log",
            "q8_decode_pair=", {"FIFI_MLX_ATTN_DEPTHS": ",".join(map(str, depth_order))},
            first, n_decode,
        )
        validate(
            records,
            [(depth, pair) for depth in depth_order for pair in range(n_decode)],
            ("len",), "fused_first", first,
            ("composite_us", "fused_us", "composite_peak_extra_bytes",
             "fused_peak_extra_bytes", "max_abs_diff"),
        )
        decode_rows.extend(
            {"process": repetition, **row, "process_peak_footprint_bytes": footprint}
            for row in records
        )
        print(f"decode repetition {repetition}: {len(records)} paired timings", flush=True)

        for rows, depth, sliced in cases if first else cases[::-1]:
            case = f"{rows}-{depth}-{sliced}"
            records, footprint = run(
                lib_binary, PREFILL_TEST, output / f"prefill-{repetition}-{case}.log",
                "dense_prefill_pair=",
                {
                    "FIFI_MLX_DENSE_BENCH_ROWS": str(rows),
                    "FIFI_MLX_DENSE_BENCH_DEPTH": str(depth),
                    "FIFI_MLX_DENSE_BENCH_SLICE": str(sliced),
                },
                first, n_prefill,
            )
            validate(
                records,
                [(rows, depth, sliced, pair) for pair in range(n_prefill)],
                ("rows", "depth", "slice"), "flash_first", first,
                ("composite_ms", "flash_ms", "composite_peak_extra_bytes",
                 "flash_peak_extra_bytes", "composite_error", "flash_error"),
            )
            prefill_rows.extend(
                {"process": repetition, **row, "process_peak_footprint_bytes": footprint}
                for row in records
            )
            print(f"prefill repetition {repetition}, {case}: {len(records)} paired timings", flush=True)
    write_csv(output / "decode.csv", decode_rows)
    write_csv(output / "prefill.csv", prefill_rows)


if __name__ == "__main__":
    main()
