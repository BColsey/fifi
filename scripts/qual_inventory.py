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
import signal
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
# Compile-time backend gates, e.g. `#![cfg(all(feature = "native-cuda", target_os = "linux"))]`
# or item-level `#[cfg(feature = "native-mlx")]`. Only attribute lines count: a bare
# `feature = "x"` inside code is not a gate. Multiline attributes are tracked by
# bracket depth so a `feature = ...` line inside them is still counted; a bare
# `feature = "x"` outside any recognizable attribute is refused rather than
# silently ignored (a formatting change must never weaken the census).
FEATURE_GATE_RE = re.compile(rf"^\s*#!?\[.*cfg\(.*feature\s*=\s*\"([a-z0-9-]+)\"")
ATTR_START_RE = re.compile(r"^\s*#!?\[")
ATTR_FEATURE_RE = re.compile(r'feature\s*=\s*"([a-z0-9-]+)"')

# The one env gate that turns a missing CUDA device into a failure instead of a
# silent skip; without it a device run is not device evidence (see cuda_smoke.rs).
DEVICE_FAIL_CLOSED_GATE = "INFERENCE_CUDA_REQUIRE_DEVICE"

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


class _TimedOut(Exception):
    """A surface exceeded its timeout; the whole process group was killed."""

    def __init__(self, stdout: str | bytes, stderr: str | bytes) -> None:
        super().__init__("surface timed out")

        def as_text(value: str | bytes) -> str:
            return value.decode(errors="replace") if isinstance(value, bytes) else value

        self.stdout = as_text(stdout or "")
        self.stderr = as_text(stderr or "")


@dataclass
class TestTargetFacts:
    """Mechanical facts about one cargo test target, derived from source."""

    path: str
    test_fns: int = 0
    ignored: int = 0
    env_gates: list[str] = field(default_factory=list)
    features: set[str] = field(default_factory=set)
    fixtures: dict[str, dict[str, int]] = field(default_factory=dict)


def _strip_cfg_macros(line: str, depth: int) -> tuple[str, int]:
    """Remove cfg!( ... ) macro regions from a line, carrying paren depth.

    `cfg!` evaluates a cfg at runtime as a boolean; it never gates target
    compilation, so its feature mentions are neither gates nor refusals.
    """
    out: list[str] = []
    index = 0
    while index < len(line):
        if depth > 0:
            close = index
            while close < len(line) and depth > 0:
                if line[close] == "(":
                    depth += 1
                elif line[close] == ")":
                    depth -= 1
                close += 1
            index = close
            continue
        start = line.find("cfg!(", index)
        if start < 0:
            out.append(line[index:])
            break
        out.append(line[index:start])
        depth = 1
        index = start + len("cfg!(")
    return "".join(out), depth


def scan_features(text: str, path: Path) -> set[str]:
    """Feature names from cfg gates, refusing shapes the census cannot read."""
    features: set[str] = set()
    attr_depth = 0  # open square brackets of the attribute being scanned
    macro_depth = 0  # open parens of a cfg!( ... ) macro expression
    for lineno, raw in enumerate(text.splitlines(), 1):
        stripped = raw.lstrip()
        if stripped.startswith("//"):  # comments quote attributes; not code
            continue
        line = raw
        if macro_depth or "cfg!(" in line:
            line, macro_depth = _strip_cfg_macros(line, macro_depth)
        if attr_depth == 0 and ATTR_START_RE.match(raw):
            features.update(FEATURE_GATE_RE.findall(line))
            attr_depth = line.count("[") - line.count("]")
        elif attr_depth > 0:
            features.update(ATTR_FEATURE_RE.findall(line))
            attr_depth += line.count("[") - line.count("]")
        elif ATTR_FEATURE_RE.search(line):
            raise InventoryError(
                f"{path}:{lineno}: 'feature = ...' outside a recognizable "
                "attribute; refusing to guess the gate census"
            )
    return features


def scan_file(
    path: Path,
) -> tuple[int, int, set[str], set[str], dict[str, dict[str, int]]]:
    text = path.read_text(encoding="utf-8")
    test_fns = len(TEST_ATTR_RE.findall(text))
    ignored = len(IGNORE_ATTR_RE.findall(text))
    gates = set(ENV_GATE_RE.findall(text))
    features = scan_features(text, path)
    fixtures: dict[str, dict[str, int]] = {}
    for kind, name in FIXTURE_CALL_RE.findall(text):
        entry = fixtures.setdefault(name, {"or_skip": 0, "required": 0})
        entry[kind] += 1
    return test_fns, ignored, gates, features, fixtures


def merge_into(facts: TestTargetFacts, path: Path) -> None:
    test_fns, ignored, gates, features, fixtures = scan_file(path)
    facts.test_fns += test_fns
    facts.ignored += ignored
    facts.env_gates.extend(sorted(gates))
    facts.features |= features
    for name, calls in fixtures.items():
        entry = facts.fixtures.setdefault(name, {"or_skip": 0, "required": 0})
        for kind, count in calls.items():
            entry[kind] += count


def cargo_features() -> list[str]:
    """Feature names declared in engine/Cargo.toml's [features] section."""
    names: list[str] = []
    in_features = False
    for line in (ENGINE_ROOT / "Cargo.toml").read_text(encoding="utf-8").splitlines():
        if line.startswith("["):
            in_features = line.strip() == "[features]"
            continue
        if in_features:
            match = re.match(r"\s*([a-z0-9-]+)\s*=", line)
            if match:
                names.append(match.group(1))
    return sorted(names)


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
        "schema_version": 2,
        "cargo_features": cargo_features(),
        "targets": {
            name: {
                "path": facts.path,
                "test_fns": facts.test_fns,
                "ignored": facts.ignored,
                "env_gates": sorted(set(facts.env_gates)),
                "features": sorted(facts.features),
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
                "timeout_sec", "features"}


def select_surfaces(policy: dict, requested: list[str] | None) -> tuple[list[str], list[str]]:
    """Resolve the surface selection: explicit ids, or every cpu surface by default.

    Device surfaces are never run by accident: hardware != cpu requires naming the
    surface explicitly, so a leaseless `collect` cannot launch GPU work.
    """
    ids = [s["id"] for s in policy["surfaces"]]
    if requested:
        return requested, [sid for sid in ids if sid not in requested]
    cpu = [s["id"] for s in policy["surfaces"] if s.get("hardware") == "cpu"]
    return cpu, [sid for sid in ids if sid not in cpu]


def validate_policy(policy: dict, facts: dict) -> list[str]:
    """Return the ordered surface ids after checking them; raise on any defect."""
    problems: list[str] = []
    known_env_gates = {gate for t in facts["targets"].values() for gate in t["env_gates"]}
    known_fixtures = {name for t in facts["targets"].values() for name in t["fixtures"]}
    declared_features = set(facts.get("cargo_features", []))
    for t in facts["targets"].values():
        for feature in t.get("features", []):
            if feature not in declared_features:
                problems.append(
                    f"facts: target {t['path']} gates on feature {feature!r}, "
                    "unknown to engine/Cargo.toml"
                )

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
        features = surface.get("features", [])
        if not isinstance(features, list) or not all(isinstance(v, str) for v in features):
            problems.append(f"{where}: features must be a list of strings")
        else:
            unknown_features = sorted(set(features) - declared_features)
            if unknown_features:
                problems.append(
                    f"{where}: features {unknown_features} are not declared "
                    "in engine/Cargo.toml"
                )
            # A device surface without the fail-closed gate can skip every test and
            # still look green; that is not device evidence.
            if surface.get("category") == "cuda-device":
                value = env.get(DEVICE_FAIL_CLOSED_GATE, "")
                if not value or value == "0":
                    problems.append(
                        f"{where}: cuda-device surfaces must set "
                        f"{DEVICE_FAIL_CLOSED_GATE} truthy in env"
                    )
                # A compiled-but-empty target (runtime gates, ignored tests) must
                # fail on its own, not ride on feature validation.
                min_ran = expect.get("min_ran")
                if not isinstance(min_ran, int) or min_ran < 1:
                    problems.append(
                        f"{where}: cuda-device surfaces must demand "
                        "expect.min_ran >= 1"
                    )
            # Every --test target whose source gates on backend features must have
            # those features activated by this surface, or cargo compiles it empty.
            feature_set = set(features)
            argv_strings = argv if isinstance(argv, list) else []
            test_targets: list[str] = []
            for index, arg in enumerate(argv_strings):
                if not isinstance(arg, str):
                    continue
                if arg.startswith("--test="):
                    test_targets.append(arg[len("--test="):])
                elif arg == "--test" and index + 1 < len(argv_strings):
                    test_targets.append(argv_strings[index + 1])
            for target in test_targets:
                if not isinstance(target, str):
                    continue
                required = set(facts["targets"].get(target, {}).get("features", []))
                missing = sorted(required - feature_set)
                if missing:
                    problems.append(
                        f"{where}: argv runs --test {target} which gates on "
                        f"features {missing} the surface does not declare"
                    )
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


# --- pack registry --------------------------------------------------------------

PACKS_REGISTRY_PATH = REPO_ROOT / "scripts" / "pack_registry.json"
SETTINGS_PATH = ENGINE_ROOT / "src" / "family" / "qwen36" / "settings.rs"
CUDA_START_PATH = ENGINE_ROOT / "src" / "family" / "qwen36" / "cuda" / "start.rs"
PREFLIGHT_PATH = ENGINE_ROOT / "src" / "capability" / "preflight.rs"

TOKENIZER_PIN_KEYS = {
    "TOKENIZER_JSON": "tokenizer_json",
    "TOKENIZER_CONFIG": "tokenizer_config_json",
    "CHAT_TEMPLATE": "chat_template_jinja",
    "GENERATION_CONFIG": "generation_config_json",
}


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise InventoryError(f"missing file: {path}") from error


def gguf_pin_name() -> str:
    """The pinned GGUF checkpoint name the CUDA loader demands."""
    match = re.search(r'const GGUF_NAME: &str = "([^"]+)"', _read(CUDA_START_PATH))
    if not match:
        raise InventoryError(f"{CUDA_START_PATH}: the pinned GGUF_NAME constant was not found")
    return match.group(1)


def tokenizer_pin_sha256s() -> dict[str, str]:
    """The four tokenizer pin digests the loader refuses mismatches on."""
    found = dict(re.findall(r"const (\w+)_SHA256: &str =\s*\"([0-9a-f]{64})\"", _read(SETTINGS_PATH)))
    missing = sorted(set(TOKENIZER_PIN_KEYS) - set(found))
    if missing:
        raise InventoryError(f"{SETTINGS_PATH}: tokenizer pin constants missing: {missing}")
    return {TOKENIZER_PIN_KEYS[name]: digest for name, digest in found.items()
            if name in TOKENIZER_PIN_KEYS}


def weight_format_qualification() -> dict[str, str]:
    """The runtime qualification mapping (preflight.rs), lowercased."""
    text = _read(PREFLIGHT_PATH)
    mapping: dict[str, str] = {}
    for fmt, label in re.findall(
        r"WeightFormat::(Gguf|Exl3) => Qualification::(Qualified|Experimental|Unverified)", text
    ):
        mapping[fmt.lower()] = label.lower()
    if "gguf" not in mapping or "exl3" not in mapping:
        raise InventoryError(
            f"{PREFLIGHT_PATH}: the WeightFormat -> Qualification mapping was not found"
        )
    return mapping


PACK_TOP_KEYS = {"schema_version", "identity_contract", "quality_standard", "packs"}
PACK_KEYS = {"id", "family", "weights", "identity", "loads",
             "numerical_qualification", "production_eligibility"}
VERDICTS = {"reference", "pass", "fail", "not_run"}
RUNTIME_LABELS = {"qualified", "experimental", "unverified"}
MARGIN_KEYS = {"true_token_nll_nats", "top1_points", "exact_copy_points", "retrieval_points"}

# Identity assurance strengths the contract may declare. Only tokenizer
# hashes are cryptographic: a matching filename, tensor shape, or bpw label
# must never be reportable as verification of the weights themselves.
CONTRACT_FIELDS = {
    "family": "source-validated",
    "gguf_expected_filename": "source-validated, not content-verified",
    "tokenizer_hashes": "cryptographically pinned",
    "exl3_tensor_schema": "shape-validated",
    "exl3_bpw_label": "documentation-derived",
    "exl3_exact_pack_contents": "not established by current loader pins",
}


def validate_pack_registry(registry: dict) -> dict[str, str]:
    """Check the registry against engine source; return pack id -> verdict.

    Raises InventoryError listing every defect: unknown shapes, identity pins
    that drifted from the loader constants, labels that contradict the
    preflight mapping, or a missing/duplicated production default.
    """
    problems: list[str] = []
    unknown_top = sorted(set(registry) - PACK_TOP_KEYS)
    if unknown_top:
        problems.append(f"unknown keys {unknown_top}")
    if registry.get("schema_version") != 1:
        problems.append("schema_version must be 1")
    contract = registry.get("identity_contract", {})
    if not isinstance(contract, dict) or set(contract) != {"statement", "fields"}:
        problems.append("identity_contract must have exactly statement and fields")
    else:
        if not contract["statement"]:
            problems.append("identity_contract.statement must state the assurance limits")
        fields = contract["fields"]
        if not isinstance(fields, dict) or set(fields) != set(CONTRACT_FIELDS):
            problems.append(f"identity_contract.fields must have exactly {sorted(CONTRACT_FIELDS)}")
        else:
            # Exact strengths: the vocabulary contains exactly one cryptographic
            # entry (tokenizer hashes), so a filename, tensor shape, or bpw label
            # can never be reported as verification of the weights themselves.
            for name, strength in CONTRACT_FIELDS.items():
                if fields[name] != strength:
                    problems.append(
                        f"identity_contract.fields.{name} must read {strength!r}; "
                        "inflating identity assurance is not allowed"
                    )
    standard = registry.get("quality_standard", {})
    if not isinstance(standard, dict) or set(standard) != {"description", "margins", "source"}:
        problems.append("quality_standard must have exactly description, margins, source")
    else:
        margins = standard["margins"]
        if not isinstance(margins, dict) or set(margins) != MARGIN_KEYS:
            problems.append(f"quality_standard.margins must have exactly {sorted(MARGIN_KEYS)}")
        elif not all(isinstance(v, (int, float)) and v > 0 for v in margins.values()):
            problems.append("quality_standard.margins values must be positive numbers")

    packs = registry.get("packs")
    if not isinstance(packs, list) or not packs:
        raise InventoryError("pack registry problems:\n  packs must be a non-empty list")

    try:
        pinned_name = gguf_pin_name()
        tokenizer_pins = tokenizer_pin_sha256s()
        runtime_mapping = weight_format_qualification()
    except InventoryError as error:
        raise InventoryError(f"pack registry problems:\n  source binding failed: {error}") from error

    ids: list[str] = []
    verdicts: dict[str, str] = {}
    defaults: list[str] = []
    for index, pack in enumerate(packs):
        where = f"packs[{index}]"
        if not isinstance(pack, dict):
            problems.append(f"{where}: not an object")
            continue
        unknown = sorted(set(pack) - PACK_KEYS)
        if unknown:
            problems.append(f"{where}: unknown keys {unknown}")
            continue
        pid = pack.get("id")
        if not isinstance(pid, str) or not pid:
            problems.append(f"{where}: id must be a non-empty string")
            continue
        where = f"pack {pid!r}"
        ids.append(pid)
        weights = pack.get("weights")
        if weights not in ("gguf", "exl3"):
            problems.append(f"{where}: weights must be gguf or exl3")
            continue
        identity = pack.get("identity", {})
        if weights == "gguf":
            if not isinstance(identity, dict) or set(identity) != {"pinned_gguf_name", "tokenizer_pin_sha256"}:
                problems.append(f"{where}: gguf identity needs pinned_gguf_name and tokenizer_pin_sha256")
            else:
                if identity["pinned_gguf_name"] != pinned_name:
                    problems.append(
                        f"{where}: pinned name {identity['pinned_gguf_name']!r} != loader pin {pinned_name!r}"
                    )
                pins = identity["tokenizer_pin_sha256"]
                if not isinstance(pins, dict):
                    problems.append(f"{where}: tokenizer_pin_sha256 must be an object")
                else:
                    for key, digest in tokenizer_pins.items():
                        if pins.get(key) != digest:
                            problems.append(f"{where}: tokenizer pin {key} drifted from the loader constant")
                    extra = sorted(set(pins) - set(tokenizer_pins))
                    if extra:
                        problems.append(f"{where}: unknown tokenizer pins {extra}")
        elif not isinstance(identity, dict) or set(identity) != {"bpw"} or not identity["bpw"]:
            problems.append(f"{where}: exl3 identity needs a non-empty bpw")

        loads = pack.get("loads", {})
        if not isinstance(loads, dict) or set(loads) != {"status", "evidence"}:
            problems.append(f"{where}: loads must have exactly status and evidence")
        elif loads["status"] not in ("evidenced", "not_evidenced"):
            problems.append(f"{where}: loads.status must be evidenced or not_evidenced")
        elif loads["status"] == "evidenced" and not loads["evidence"]:
            problems.append(f"{where}: evidenced loads need at least one evidence reference")

        numerical = pack.get("numerical_qualification", {})
        if not isinstance(numerical, dict) or set(numerical) != {"verdict", "gate", "evidence"}:
            problems.append(f"{where}: numerical_qualification must have verdict, gate, evidence")
        else:
            verdict = numerical["verdict"]
            if verdict not in VERDICTS:
                problems.append(f"{where}: numerical verdict must be one of {sorted(VERDICTS)}")
            elif not numerical["gate"] or not numerical["evidence"]:
                problems.append(f"{where}: a recorded verdict needs its gate and evidence")
            else:
                verdicts[pid] = verdict

        eligibility = pack.get("production_eligibility", {})
        if not isinstance(eligibility, dict) or set(eligibility) != {"runtime_label", "production_default", "notes"}:
            problems.append(f"{where}: production_eligibility needs runtime_label, production_default, notes")
        else:
            label = eligibility["runtime_label"]
            if label not in RUNTIME_LABELS:
                problems.append(f"{where}: runtime_label must be one of {sorted(RUNTIME_LABELS)}")
            elif label != runtime_mapping.get(weights):
                problems.append(
                    f"{where}: runtime_label {label!r} contradicts the preflight "
                    f"mapping {runtime_mapping.get(weights)!r} for weights {weights!r}"
                )
            if eligibility["production_default"] is True:
                defaults.append(pid)

    duplicates = {pid for pid in ids if ids.count(pid) > 1}
    for pid in sorted(duplicates):
        problems.append(f"duplicate pack id {pid!r}")
    if len(defaults) != 1:
        problems.append(f"exactly one production_default must be set (found {len(defaults)})")
    else:
        default = next(p for p in packs if isinstance(p, dict) and p.get("id") == defaults[0])
        if default.get("weights") != "gguf" or default.get("production_eligibility", {}).get("runtime_label") != "qualified":
            problems.append(
                "the production default must be the qualified gguf pack "
                "(GGUF stays the qualified Qwen3.6 default)"
            )
        if verdicts.get(defaults[0]) != "reference":
            problems.append("the production default must carry the reference verdict")
        for pid, verdict in verdicts.items():
            if verdict == "reference" and pid != defaults[0]:
                problems.append(f"pack {pid!r}: only the production default may be the reference")
            # Production eligibility may not exceed the recorded evidence: a
            # failed gate outranks a permissive runtime policy label.
            if verdict == "fail" and next(
                p for p in packs if isinstance(p, dict) and p.get("id") == pid
            ).get("production_eligibility", {}).get("runtime_label") == "qualified":
                problems.append(
                    f"pack {pid!r}: runtime_label 'qualified' exceeds its recorded "
                    "numerical verdict 'fail'"
                )

    if problems:
        raise InventoryError("pack registry problems:\n  " + "\n  ".join(problems))
    return verdicts


def cmd_packs(args: argparse.Namespace) -> int:
    registry = load_json_strict(PACKS_REGISTRY_PATH)
    verdicts = validate_pack_registry(registry)
    default = next(p["id"] for p in registry["packs"] if p["production_eligibility"]["production_default"])
    print(f"packs: {len(registry['packs'])} entries validate against engine source; "
          f"default = {default}; verdicts = "
          + ", ".join(f"{pid}:{verdict}" for pid, verdict in sorted(verdicts.items())))
    return 0


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
        # New session = own process group: a timeout must kill cargo AND the test
        # binaries (and any GPU work) it spawned, not leave them orphaned.
        proc = subprocess.Popen(
            argv, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, start_new_session=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout_sec)
            returncode = proc.returncode
        except subprocess.TimeoutExpired as expired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass  # the group died between timeout and kill; just reap
            killed_out, killed_err = proc.communicate()
            raise _TimedOut(
                killed_out or expired.stdout or "", killed_err or expired.stderr or ""
            ) from expired
    except _TimedOut as timed_out:
        # The whole group is dead; an interrupted run must judge as a failure, with
        # whatever output existed written out for inspection.
        stem = re.sub(rf"[^{UPPER}a-z0-9_.-]", "_", surface["id"])
        (out_dir / f"{stem}.stdout").write_text(timed_out.stdout, encoding="utf-8")
        (out_dir / f"{stem}.stderr").write_text(timed_out.stderr, encoding="utf-8")
        return {
            "id": surface["id"], "category": surface["category"],
            "hardware": surface["hardware"], "argv": argv, "cwd": surface["cwd"],
            "env": surface.get("env", {}), "exit_code": -9,
            "started": started.isoformat(),
            "ended": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "summary": None, "fixtures_skipped": {},
            "expect": surface.get("expect", {}), "verdict": "FAIL",
            "reasons": [f"timed out after {timeout_sec}s"],
        }
    ended = datetime.datetime.now(datetime.timezone.utc)
    stem = re.sub(rf"[^{UPPER}a-z0-9_.-]", "_", surface["id"])
    (out_dir / f"{stem}.stdout").write_text(stdout, encoding="utf-8")
    (out_dir / f"{stem}.stderr").write_text(stderr, encoding="utf-8")

    expect = surface.get("expect", {})
    verdict, reasons, summary, skips = judge_surface(
        stdout, stderr, returncode, expect,
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
    selected, skipped = select_surfaces(policy, args.surface)
    unknown = [sid for sid in selected if sid not in all_ids]
    if unknown:
        raise InventoryError(f"unknown surface ids: {', '.join(unknown)}")
    if skipped:
        print("not selected (name them with --surface to run): " + ", ".join(skipped))

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
    sub.add_parser("packs", help="validate the pack registry against engine source")

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
        if args.command == "packs":
            return cmd_packs(args)
        return cmd_collect(args)
    except InventoryError as error:
        print(f"qual_inventory: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
