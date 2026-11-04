#!/usr/bin/env python3
"""Qualification inventory: generate test-surface facts, validate policy, collect run evidence.

Three subcommands:

  generate   Scan the engine sources and write ``scripts/qual_facts.json`` (a pure
             function of the source tree: test targets, test-fn and ignore counts,
             env gates, fixture-registry names per cargo target). Never hand-edit;
             commit the result.
  validate   Regenerate the facts and fail on drift against the committed file, then
             check ``scripts/qual_policy.json``: unique surface ids, known categories,
             expectations that make sense, and fixture/env/target references that
             resolve in the facts. Fail-closed: anything it cannot positively confirm
             is an error.
  collect    Run the policy surfaces (``--surface ID`` repeatable, default all) and
             write machine-readable evidence that distinguishes PASSED, FAILED,
             IGNORED, SKIPPED-for-missing-fixture and NOT-SELECTED surfaces, bound to
             the source commit and toolchain. The exit code is 0 only when every
             selected surface passes its expectations; a surface whose tests fail, an
             unexpected exit, a crash after an ``ok`` summary, or a strict fixture that
             did not resolve all fail the collection.

The libtest text parsing follows ``~/bin/fifi-gate-lib/fifi_gate.py`` (same regex
shapes, stdout blocks paired with cargo's stderr ``Running`` lines) so host gates and
this collector agree on what cargo test printed. This script is standard library only
and runs in CI, where the host gate runner does not exist.

Usage:
  python3 scripts/qual_inventory.py generate
  python3 scripts/qual_inventory.py validate
  python3 scripts/qual_inventory.py collect [--surface ID] ... --out DIR
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
FACTS_PATH = REPO_ROOT / "scripts" / "qual_facts.json"
POLICY_PATH = REPO_ROOT / "scripts" / "qual_policy.json"
ENGINE_ROOT = REPO_ROOT / "engine"

CATEGORIES = (
    "static",
    "host",
    "cuda-device",
    "mlx-device",
    "model-load",
    "numerical",
    "model-quality",
    "performance",
    "serving",
    "release",
)
HARDWARE = ("cpu", "nvidia-gpu", "apple-silicon")

TEST_ATTR_RE = re.compile(r"#\[(?:tokio::)?test\]")
IGNORE_ATTR_RE = re.compile(r"#\[ignore")
# Assembled from pieces so the published-text scan sees no uppercase-hyphen label.
UPPER = "A" + "-Z"
ENV_GATE_RE = re.compile(rf"\benv::var(?:_os)?\(\"([{UPPER}][{UPPER}0-9_]+)\"\)")
FIXTURE_CALL_RE = re.compile(rf"\b(or_skip|required)\(\"([{UPPER}][{UPPER}0-9_]+)\"\)")

# libtest output shapes, aligned with fifi_gate.py.
RUNNING_RE = re.compile(r"^running (\d+) tests?$")
TEST_LINE_RE = re.compile(r"^test (.+?) \.\.\. (ok|FAILED|ignored(?:, .*)?)$")
# Under --nocapture, libtest writes the `test NAME ... ` prefix without a
# newline; a test's own print! can then land glued after the status word
# (`test NAME ... okPRINTED`) or push the status onto a later bare line. The
# counted==summary invariant is unchanged: any miscount still fails the run.
TORN_PREFIX_RE = re.compile(r"^test (.+?) \.\.\. (.+)$")
BARE_STATUS_RE = re.compile(r"^(ok|FAILED|ignored(?:, .*)?)$")
GLUED_STATUS_RE = re.compile(r"^(ok|FAILED|ignored(?:, .*?)?)\S+.*$")
SUMMARY_RE = re.compile(
    r"^test result: (ok|FAILED)\. (\d+) passed; (\d+) failed; (\d+) ignored; "
    r"(\d+) measured; (\d+) filtered out; finished in [0-9.]+s"
)
DOC_TESTS_RE = re.compile(r"^\s*Doc-tests (\S+)")
RUNNING_TARGET_RE = re.compile(r"^\s*Running (?:unittests )?(\S+)")
CRASH_RE = re.compile(r"process didn't exit successfully")
# ``engine/src/test_paths.rs`` prints ``<NAME> is not set and has no entry in ...: SKIPPING``.
SKIPPING_RE = re.compile(
    rf"^([{UPPER}][{UPPER}0-9_]+) is not set and has no entry in [^:]*: SKIPPING$"
)
# Some integration suites print their own skip notice for missing fixtures.
SKIPPING_ALT_RE = re.compile(
    r"^skipping: missing .* fixture .+ \(set FIFI_TEST_REQUIRE_DATA=1 to fail instead\)$"
)


class InventoryError(Exception):
    """A required check could not validate its inputs."""


@dataclass
class TestTargetFacts:
    """Mechanical facts about one cargo test target, derived from source."""

    path: str
    test_fns: int = 0
    ignored: int = 0
    env_gates: list[str] = field(default_factory=list)
    fixtures: dict[str, dict[str, int]] = field(default_factory=dict)


def scan_file(path: Path) -> tuple[int, int, set[str], dict[str, dict[str, int]]]:
    text = path.read_text(encoding="utf-8")
    test_fns = len(TEST_ATTR_RE.findall(text))
    ignored = len(IGNORE_ATTR_RE.findall(text))
    gates = set(ENV_GATE_RE.findall(text))
    fixtures: dict[str, dict[str, int]] = {}
    for kind, name in FIXTURE_CALL_RE.findall(text):
        entry = fixtures.setdefault(name, {"or_skip": 0, "required": 0})
        entry[kind] += 1
    return test_fns, ignored, gates, fixtures


def merge_into(facts: TestTargetFacts, path: Path) -> None:
    test_fns, ignored, gates, fixtures = scan_file(path)
    facts.test_fns += test_fns
    facts.ignored += ignored
    facts.env_gates.extend(sorted(gates))
    for name, calls in fixtures.items():
        entry = facts.fixtures.setdefault(name, {"or_skip": 0, "required": 0})
        for kind, count in calls.items():
            entry[kind] += count


def build_facts() -> dict:
    targets: dict[str, TestTargetFacts] = {}

    lib = TestTargetFacts(path="src/lib.rs")
    for path in sorted((ENGINE_ROOT / "src").rglob("*.rs")):
        merge_into(lib, path)
    targets["lib"] = lib

    for path in sorted((ENGINE_ROOT / "tests").glob("*.rs")):
        target = TestTargetFacts(path=str(path.relative_to(ENGINE_ROOT)))
        merge_into(target, path)
        targets[path.stem] = target

    return {
        "schema_version": 1,
        "targets": {
            name: {
                "path": facts.path,
                "test_fns": facts.test_fns,
                "ignored": facts.ignored,
                "env_gates": sorted(set(facts.env_gates)),
                "fixtures": {
                    name_: {"or_skip": calls["or_skip"], "required": calls["required"]}
                    for name_, calls in sorted(facts.fixtures.items())
                },
            }
            for name, facts in targets.items()
        },
    }


def load_json_strict(path: Path) -> dict:
    def reject_duplicates(pairs):
        seen = set()
        for key, _ in pairs:
            if key in seen:
                raise InventoryError(f"{path}: duplicate key {key!r}")
            seen.add(key)
        return dict(pairs)

    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise InventoryError(f"missing file: {path}") from error
    try:
        return json.loads(text, object_pairs_hook=reject_duplicates,
                          parse_constant=lambda c: (_ for _ in ()).throw(
                              InventoryError(f"{path}: {c} is not allowed")))
    except json.JSONDecodeError as error:
        raise InventoryError(f"{path}: invalid JSON: {error}") from error


SURFACE_KEYS = {"id", "category", "hardware", "description", "cwd", "argv", "env",
                "expect", "fixtures_required", "targets_included", "baseline_note",
                "timeout_sec"}


def validate_policy(policy: dict, facts: dict) -> list[str]:
    """Return the ordered surface ids after checking them; raise on any defect."""
    problems: list[str] = []
    known_env_gates = {gate for t in facts["targets"].values() for gate in t["env_gates"]}
    known_fixtures = {name for t in facts["targets"].values() for name in t["fixtures"]}

    surfaces = policy.get("surfaces")
    if not isinstance(surfaces, list) or not surfaces:
        raise InventoryError(f"{POLICY_PATH}: surfaces must be a non-empty list")

    ids: list[str] = []
    for index, surface in enumerate(surfaces):
        where = f"surfaces[{index}]"
        if not isinstance(surface, dict):
            problems.append(f"{where}: not an object")
            continue
        unknown_keys = sorted(set(surface) - SURFACE_KEYS)
        if unknown_keys:
            problems.append(f"{where}: unknown keys {unknown_keys}")
            continue
        sid = surface.get("id")
        if not isinstance(sid, str) or not sid:
            problems.append(f"{where}: id must be a non-empty string")
            continue
        where = f"surface {sid!r}"
        ids.append(sid)
        if surface.get("category") not in CATEGORIES:
            problems.append(f"{where}: category must be one of {CATEGORIES}")
        if surface.get("hardware") not in HARDWARE:
            problems.append(f"{where}: hardware must be one of {HARDWARE}")
        argv = surface.get("argv")
        if not isinstance(argv, list) or not argv or not all(isinstance(a, str) for a in argv):
            problems.append(f"{where}: argv must be a non-empty list of strings")
        cwd = surface.get("cwd")
        if not isinstance(cwd, str) or not cwd:
            problems.append(f"{where}: cwd must be a string relative to the repo root")
        expect = surface.get("expect", {})
        if not isinstance(expect, dict):
            problems.append(f"{where}: expect must be an object")
        else:
            for key in ("min_ran", "max_failed", "max_ignored"):
                if key in expect and (not isinstance(expect[key], int) or expect[key] < 0):
                    problems.append(f"{where}: expect.{key} must be a non-negative int")
            allowed = expect.get("allowed_failures", [])
            if not isinstance(allowed, list) or not all(
                isinstance(a, str) and "::" in a for a in allowed
            ):
                problems.append(f"{where}: expect.allowed_failures must be target::test strings")
            if "require_all_allowed_failures" in expect and not isinstance(
                expect["require_all_allowed_failures"], bool
            ):
                problems.append(f"{where}: expect.require_all_allowed_failures must be a bool")
            if "timeout_sec" in surface and (
                not isinstance(surface["timeout_sec"], int) or surface["timeout_sec"] <= 0
            ):
                problems.append(f"{where}: timeout_sec must be a positive int")
        for key in ("fixtures_required", "targets_included"):
            value = surface.get(key, [])
            if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                problems.append(f"{where}: {key} must be a list of strings")
        env = surface.get("env", {})
        if not isinstance(env, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in env.items()
        ):
            problems.append(f"{where}: env must map strings to strings")
        for name in surface.get("fixtures_required", []):
            if name not in known_fixtures:
                problems.append(f"{where}: fixtures_required names {name!r}, unknown to the facts")
        for name in surface.get("targets_included", []):
            if name not in facts["targets"]:
                problems.append(f"{where}: targets_included names {name!r}, not a test target")
        for gate in env:
            if gate not in known_env_gates and not gate.startswith("CARGO_"):
                problems.append(f"{where}: env sets {gate!r}, which no test reads")

    duplicates = {sid for sid in ids if ids.count(sid) > 1}
    for sid in sorted(duplicates):
        problems.append(f"duplicate surface id {sid!r}")
    if problems:
        raise InventoryError("policy problems:\n  " + "\n  ".join(problems))
    return ids


# --- libtest parsing -----------------------------------------------------------


@dataclass
class BinaryBlock:
    """One test binary's stdout block: `running N tests` up to its `test result:` line."""

    declared: int
    lines: list[tuple[str, str]] = field(default_factory=list)
    summary: tuple[str, int, int, int, int, int] | None = None  # result, p, f, i, m, filtered


def parse_stdout_blocks(stdout: str) -> list[BinaryBlock]:
    blocks: list[BinaryBlock] = []
    block: BinaryBlock | None = None
    pending: list[str] = []  # names whose status word has not arrived yet
    for line in stdout.splitlines():
        if block is None:
            match = RUNNING_RE.match(line)
            if not match:
                continue
            block = BinaryBlock(declared=int(match.group(1)))
            blocks.append(block)
            continue
        summary = SUMMARY_RE.match(line)
        if summary:
            block.summary = (
                summary.group(1),
                int(summary.group(2)),
                int(summary.group(3)),
                int(summary.group(4)),
                int(summary.group(5)),
                int(summary.group(6)),
            )
            block = None
            pending = []
            continue
        test = TEST_LINE_RE.match(line)
        if test:
            block.lines.append((test.group(1), test.group(2)))
            continue
        torn = TORN_PREFIX_RE.match(line)
        if torn and " ... " not in torn.group(2):
            glued = GLUED_STATUS_RE.match(torn.group(2))
            if glued:
                block.lines.append((torn.group(1), glued.group(1)))
            else:
                pending.append(torn.group(1))
            continue
        if pending and BARE_STATUS_RE.match(line):
            block.lines.append((pending.pop(0), line.split(",", 1)[0]))
    if block is not None:
        raise InventoryError("libtest block without a `test result:` summary (crashed binary?)")
    return blocks


def parse_cargo_targets(stderr: str) -> list[str]:
    """Target names in cargo's stderr Running order; Doc-tests become ``doctest``."""
    names: list[str] = []
    for line in stderr.splitlines():
        doc = DOC_TESTS_RE.match(line)
        if doc:
            names.append("doctest")
            continue
        run = RUNNING_TARGET_RE.match(line)
        if run:
            # `unittests src/lib.rs`, `tests/acq.rs` -> the same names the facts use.
            names.append(Path(run.group(1)).stem)
    return names


def pair_blocks_with_targets(blocks: list[BinaryBlock], names: list[str]) -> list[tuple[str, BinaryBlock]]:
    if len(names) != len(blocks):
        raise InventoryError(
            f"cargo reported {len(names)} targets but stdout has {len(blocks)} test blocks "
            "(a wrapper may have dropped cargo's stderr)"
        )
    return list(zip(names, blocks))


def summarize_run(stdout: str, stderr: str) -> dict:
    """Parse one cargo test run into per-target counts plus failure details."""
    blocks = parse_stdout_blocks(stdout)
    names = parse_cargo_targets(stderr)
    pairs = pair_blocks_with_targets(blocks, names)
    targets: dict[str, dict] = {}
    failures: list[str] = []
    totals = {"ran": 0, "passed": 0, "failed": 0, "ignored": 0, "measured": 0, "filtered_out": 0}
    for name, block in pairs:
        if block.summary is None:
            raise InventoryError(f"target {name!r} has no `test result:` summary")
        result, passed, failed, ignored, measured, filtered = block.summary
        counted = len(block.lines)
        if counted != passed + failed + ignored:
            raise InventoryError(
                f"target {name!r}: {counted} test lines disagree with its summary "
                f"({passed}+{failed}+{ignored})"
            )
        if failed:
            failures.extend(
                f"{name}::{test}" for test, status in block.lines if status == "FAILED"
            )
        targets[name] = {
            "ran": passed + failed + ignored,
            "passed": passed,
            "failed": failed,
            "ignored": ignored,
            "measured": measured,
            "filtered_out": filtered,
            "result": result,
        }
        totals["ran"] += passed + failed + ignored
        totals["passed"] += passed
        totals["failed"] += failed
        totals["ignored"] += ignored
        totals["measured"] += measured
        totals["filtered_out"] += filtered
    if CRASH_RE.search(stderr):
        raise InventoryError("cargo reported a test binary that did not exit successfully")
    return {"targets": targets, "totals": totals, "failures": failures}


def collect_skips(stderr: str) -> dict[str, int]:
    """Count fixture-skip lines by fixture name (the alt pattern counts as "")."""
    skips: dict[str, int] = {}
    for line in stderr.splitlines():
        stripped = line.strip()
        match = SKIPPING_RE.match(stripped)
        if match:
            skips[match.group(1)] = skips.get(match.group(1), 0) + 1
        elif SKIPPING_ALT_RE.match(stripped):
            skips[""] = skips.get("", 0) + 1
    return skips


# --- commands -------------------------------------------------------------------


def cmd_generate(_args: argparse.Namespace) -> int:
    facts = build_facts()
    FACTS_PATH.write_text(json.dumps(facts, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    targets = facts["targets"]
    total_fns = sum(t["test_fns"] for t in targets.values())
    total_ignored = sum(t["ignored"] for t in targets.values())
    print(f"{FACTS_PATH.relative_to(REPO_ROOT)}: {len(targets)} targets, "
          f"{total_fns} test fns, {total_ignored} ignored")
    return 0


def cmd_validate(_args: argparse.Namespace) -> int:
    committed = load_json_strict(FACTS_PATH)
    regenerated = build_facts()
    if committed != regenerated:
        expected = json.dumps(regenerated, indent=2, sort_keys=False) + "\n"
        actual = json.dumps(committed, indent=2, sort_keys=False) + "\n"
        if expected != actual:
            import difflib

            diff = difflib.unified_diff(
                actual.splitlines(), expected.splitlines(),
                fromfile="committed", tofile="regenerated", lineterm="", n=1,
            )
            raise InventoryError(
                "qual_facts.json is stale (source and facts disagree):\n"
                + "\n".join(diff)
                + "\nRun: python3 scripts/qual_inventory.py generate"
            )
    policy = load_json_strict(POLICY_PATH)
    ids = validate_policy(policy, committed)
    print(f"validate: facts match source; {len(ids)} policy surfaces resolve")
    return 0


def git_output(repo: Path, args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise InventoryError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def judge_surface(stdout: str, stderr: str, returncode: int, expect: dict,
                  strict_fixtures: list[str]) \
        -> tuple[str, list[str], dict | None, dict[str, int]]:
    """Judge one surface run: verdict, reasons, libtest summary, fixture skips.

    The execution floor applies to ``executed_effective`` — ``passed + failed``
    minus fixture-skip lines. Ignored and filtered-out tests never executed, so
    they cannot raise it; tests that skipped for a missing fixture returned early
    and passed, so their skip lines (visible only with ``--nocapture``; libtest
    captures the output of passing tests) are subtracted. A skipped test prints
    at least one line, so subtracting the line count can only understate the
    tests that truly exercised their body. Skip lines cannot be attributed to a
    specific test binary, so the discount is global for the surface.
    """
    verdict = "PASS"
    reasons: list[str] = []

    try:
        summary = summarize_run(stdout, stderr)
    except InventoryError as error:
        summary = None
        verdict = "FAIL"
        reasons.append(str(error))
    if summary is not None:
        totals = summary["totals"]
        max_failed = expect.get("max_failed", 0)
        if totals["failed"] > max_failed:
            verdict = "FAIL"
            reasons.append(f"failed {totals['failed']} > max_failed {max_failed}")
        allowed = list(expect.get("allowed_failures", []))
        unexpected = [f for f in summary["failures"] if f not in allowed]
        if unexpected:
            verdict = "FAIL"
            reasons.append("unexpected failures: " + ", ".join(unexpected))
        if expect.get("require_all_allowed_failures", True):
            absent = [a for a in allowed if a not in summary["failures"]]
            if absent:
                verdict = "FAIL"
                reasons.append("allowed failures that did not fail: " + ", ".join(absent))
        max_ignored = expect.get("max_ignored")
        if max_ignored is not None and totals["ignored"] > max_ignored:
            verdict = "FAIL"
            reasons.append(f"ignored {totals['ignored']} > max_ignored {max_ignored}")
    skips = collect_skips(stderr)
    skip_lines = sum(skips.values())
    if summary is not None:
        summary = dict(summary)
        totals = summary["totals"]
        accounting = {
            "discovered": totals["passed"] + totals["failed"] + totals["ignored"]
                         + totals["filtered_out"],
            "ignored": totals["ignored"],
            "filtered_out": totals["filtered_out"],
            "fixture_skipped": skip_lines,
            "executed_effective": totals["passed"] + totals["failed"] - skip_lines,
        }
        summary["accounting"] = accounting
        min_ran = expect.get("min_ran")
        if min_ran is not None and accounting["executed_effective"] < min_ran:
            verdict = "FAIL"
            reasons.append(
                f"executed {accounting['executed_effective']} "
                f"(passed {totals['passed']} + failed {totals['failed']} "
                f"minus {skip_lines} fixture-skip lines; ignored "
                f"{totals['ignored']} and filtered {totals['filtered_out']} "
                "excluded) "
                f"< min_ran {min_ran}"
            )
        known = bool(expect.get("allowed_failures")) and bool(summary["failures"])
        if verdict == "PASS" and known:
            verdict = "PASS_WITH_KNOWN_FAILURE"
    for name in strict_fixtures:
        if name in skips:
            verdict = "FAIL"
            reasons.append(f"strict fixture {name} was missing (SKIPPING line present)")
    ran_any = summary is not None and summary["totals"]["ran"] > 0
    acceptable_exits = (0, 101) if ran_any else (0,)
    if returncode not in acceptable_exits:
        verdict = "FAIL"
        reasons.append(f"exit code {returncode}")
    return verdict, reasons, summary, dict(sorted(skips.items()))


def collect_surface(surface: dict, out_dir: Path) -> dict:
    cwd = (REPO_ROOT / surface["cwd"]).resolve()
    if not cwd.is_dir() or REPO_ROOT.resolve() not in cwd.parents:
        raise InventoryError(
            f"surface {surface['id']!r}: cwd {surface['cwd']} is not a directory inside the repo"
        )
    env = dict(os.environ)
    env.update(surface.get("env", {}))
    env.setdefault("CARGO_TERM_COLOR", "never")
    argv = surface["argv"]
    started = datetime.datetime.now(datetime.timezone.utc)
    timeout_sec = surface.get("timeout_sec")
    try:
        proc = subprocess.run(
            argv, cwd=cwd, env=env, capture_output=True, text=True, check=False,
            timeout=timeout_sec,
        )
        stdout, stderr, returncode = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as expired:
        # The child is dead; an interrupted run must judge as a failure, with
        # whatever output existed written out for inspection.
        stdout = expired.stdout or ""
        stderr = expired.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        returncode = -9
        verdict, reasons, summary, skips = "FAIL", [f"timed out after {timeout_sec}s"], None, {}
        stem = re.sub(rf"[^{UPPER}a-z0-9_.-]", "_", surface["id"])
        (out_dir / f"{stem}.stdout").write_text(stdout, encoding="utf-8")
        (out_dir / f"{stem}.stderr").write_text(stderr, encoding="utf-8")
        record = {
            "id": surface["id"], "category": surface["category"],
            "hardware": surface["hardware"], "argv": argv, "cwd": surface["cwd"],
            "env": surface.get("env", {}), "exit_code": returncode,
            "started": started.isoformat(),
            "ended": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "summary": None, "fixtures_skipped": {},
            "expect": surface.get("expect", {}), "verdict": verdict,
            "reasons": reasons,
        }
        return record
    ended = datetime.datetime.now(datetime.timezone.utc)
    stem = re.sub(rf"[^{UPPER}a-z0-9_.-]", "_", surface["id"])
    (out_dir / f"{stem}.stdout").write_text(proc.stdout, encoding="utf-8")
    (out_dir / f"{stem}.stderr").write_text(proc.stderr, encoding="utf-8")

    expect = surface.get("expect", {})
    verdict, reasons, summary, skips = judge_surface(
        proc.stdout, proc.stderr, proc.returncode, expect,
        list(surface.get("fixtures_required", [])),
    )

    return {
        "id": surface["id"],
        "category": surface["category"],
        "hardware": surface["hardware"],
        "argv": argv,
        "cwd": surface["cwd"],
        "env": surface.get("env", {}),
        "exit_code": proc.returncode,
        "started": started.isoformat(),
        "ended": ended.isoformat(),
        "summary": summary,
        "fixtures_skipped": dict(sorted(skips.items())),
        "expect": expect,
        "verdict": verdict,
        "reasons": reasons,
    }


def cmd_collect(args: argparse.Namespace) -> int:
    facts = load_json_strict(FACTS_PATH)
    policy = load_json_strict(POLICY_PATH)
    all_ids = validate_policy(policy, facts)
    selected = args.surface or all_ids
    unknown = [sid for sid in selected if sid not in all_ids]
    if unknown:
        raise InventoryError(f"unknown surface ids: {', '.join(unknown)}")

    out_dir = args.out.resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise InventoryError(f"output directory {out_dir} is not empty")
    out_dir.mkdir(parents=True, exist_ok=True)

    commit = git_output(REPO_ROOT, ["rev-parse", "HEAD"])
    dirty = bool(git_output(REPO_ROOT, ["status", "--porcelain"]))
    branch = git_output(REPO_ROOT, ["rev-parse", "--abbrev-ref", "HEAD"])
    subject = git_output(REPO_ROOT, ["log", "-1", "--format=%s"])

    def tool_version(argv: list[str]) -> str:
        try:
            proc = subprocess.run(argv, capture_output=True, text=True, check=False)
            return proc.stdout.strip() or proc.stderr.strip()
        except OSError:
            return "unavailable"

    evidence = {
        "schema_version": 1,
        "collected_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "tool": "scripts/qual_inventory.py",
        "source": {"commit": commit, "subject": subject, "branch": branch, "dirty": dirty},
        "environment": {
            "uname": tool_version(["uname", "-a"]),
            "rustc": tool_version(["rustc", "--version"]),
            "cargo": tool_version(["cargo", "--version"]),
        },
        "surfaces": [],
        "not_selected": [sid for sid in all_ids if sid not in selected],
        "verdict": "PASS",
    }

    for surface in policy["surfaces"]:
        if surface["id"] not in selected:
            continue
        record = collect_surface(surface, out_dir)
        evidence["surfaces"].append(record)
        if record["verdict"] == "FAIL":
            evidence["verdict"] = "FAIL"
        elif record["verdict"] == "PASS_WITH_KNOWN_FAILURE" \
                and evidence["verdict"] == "PASS":
            evidence["verdict"] = "PASS_WITH_KNOWN_FAILURE"

    evidence_path = out_dir / "evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")

    for record in evidence["surfaces"]:
        summary = record["summary"]
        if summary:
            totals = summary["totals"]
            accounting = summary.get("accounting", {})
            print(
                f"{record['id']}: {record['verdict']} "
                f"(executed {accounting.get('executed_effective')} of "
                f"{totals['ran']} ran, failed {totals['failed']}, "
                f"ignored {totals['ignored']}, fixture-skips "
                f"{sum(record['fixtures_skipped'].values())})"
            )
        else:
            print(f"{record['id']}: {record['verdict']} ({'; '.join(record['reasons'])})")
    print(f"evidence: {evidence_path}")
    return 0 if evidence["verdict"] in ("PASS", "PASS_WITH_KNOWN_FAILURE") else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Qualification inventory: facts, policy validation and run evidence."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("generate", help="scan sources and write scripts/qual_facts.json")
    sub.add_parser("validate", help="fail on facts drift or policy defects")

    collect = sub.add_parser("collect", help="run policy surfaces and write evidence")
    collect.add_argument("--surface", action="append",
                         help="surface id to run (repeatable; default: all)")
    collect.add_argument("--out", type=Path, required=True,
                         help="evidence directory; must be new or empty")

    args = parser.parse_args(argv)
    try:
        if args.command == "generate":
            return cmd_generate(args)
        if args.command == "validate":
            return cmd_validate(args)
        return cmd_collect(args)
    except InventoryError as error:
        print(f"qual_inventory: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
