"""Offline tests for the qualification inventory script: census, parsing, and validation.

The collector's process running (cargo, git) is exercised by real runs recorded in
the task evidence; these tests pin the pure logic that makes evidence trustworthy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import qual_inventory as qi  # noqa: E402


GOOD_BLOCK = """running 3 tests
test alpha ... ok
test beta ... ignored, parent-owned GPU window only
test gamma ... FAILED
test result: FAILED. 1 passed; 1 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.01s
"""

OK_BLOCK = """running 2 tests
test one ... ok
test two ... ok
test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s
"""

CARGO_STDERR = """   Compiling inference-engine v0.1.0
    Finished test [unoptimized + debuginfo] target(s) in 3.00s
     Running unittests src/lib.rs (target/debug/deps/inference_engine-abc)
     Running tests/acquisition.rs (target/debug/deps/acquisition-def)
     Doc-tests inference_engine
"""


def parse_run(stdout: str, stderr: str) -> dict:
    return qi.summarize_run(stdout, stderr)


class ScanFileTest(unittest.TestCase):
    def test_counts_tests_ignores_gates_and_fixtures(self) -> None:
        source = """\
#[test]
fn a() {}

#[tokio::test]
#[ignore = "device only"]
async fn b() {}

#[test]
fn c() {
    let d = crate::test_paths::required("Q36_CUDA_MODEL_DIR");
    let s = test_paths::or_skip("FIFI_QWEN36_MODEL_DIR");
    let g = std::env::var("FLINT_QUAL_GO");
    let h = std::env::var_os("FIFI_FLINT_GGUF");
}
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.rs"
            path.write_text(source, encoding="utf-8")
            fns, ignored, gates, fixtures = qi.scan_file(path)
        self.assertEqual(fns, 3)
        self.assertEqual(ignored, 1)
        self.assertEqual(gates, {"FLINT_QUAL_GO", "FIFI_FLINT_GGUF"})
        self.assertEqual(
            fixtures,
            {
                "Q36_CUDA_MODEL_DIR": {"or_skip": 0, "required": 1},
                "FIFI_QWEN36_MODEL_DIR": {"or_skip": 1, "required": 0},
            },
        )


class SummarizeRunTest(unittest.TestCase):
    def test_totals_targets_and_failures(self) -> None:
        summary = parse_run(GOOD_BLOCK + OK_BLOCK,
                            "     Running unittests src/lib.rs (x)\n"
                            "     Running tests/acq.rs (y)\n")
        self.assertEqual(summary["totals"],
                         {"ran": 5, "passed": 3, "failed": 1, "ignored": 1,
                          "measured": 0, "filtered_out": 0})
        self.assertEqual(sorted(summary["targets"]), ["acq", "lib"])
        self.assertEqual(summary["targets"]["lib"]["failed"], 1)
        self.assertEqual(summary["failures"], ["lib::gamma"])

    def test_doctests_paired_after_binaries(self) -> None:
        summary = parse_run(OK_BLOCK + OK_BLOCK + OK_BLOCK, CARGO_STDERR)
        self.assertEqual(sorted(summary["targets"]), ["acquisition", "doctest", "lib"])

    def test_target_count_mismatch_fails(self) -> None:
        stderr = ("     Running unittests src/lib.rs (x)\n"
                  "     Running tests/acq.rs (y)\n")
        with self.assertRaises(qi.InventoryError):
            parse_run(OK_BLOCK, stderr)

    def test_line_summary_disagreement_fails(self) -> None:
        bad = "running 2 tests\ntest one ... ok\ntest result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.0s\n"
        with self.assertRaises(qi.InventoryError):
            parse_run(bad, "     Running unittests src/lib.rs (x)\n")

    def test_block_without_summary_fails(self) -> None:
        with self.assertRaises(qi.InventoryError):
            qi.parse_stdout_blocks("running 1 tests\ntest one ... ok\n")

    def test_crash_after_ok_fails(self) -> None:
        stderr = "     Running unittests src/lib.rs (x)\nprocess didn't exit successfully"
        with self.assertRaises(qi.InventoryError):
            parse_run(OK_BLOCK, stderr)

    def test_empty_output_with_floor_reported_by_caller(self) -> None:
        # A compile failure prints no blocks; summarize succeeds with zero totals,
        # so the caller's min_ran floor is what fails the surface.
        summary = parse_run("", "")
        self.assertEqual(summary["totals"]["ran"], 0)


class SkipsTest(unittest.TestCase):
    def test_skipping_lines_collected_by_name(self) -> None:
        stderr = ("FIFI_QWEN36_MODEL_DIR is not set and has no entry in ~/.config/fifi/test-paths.json: SKIPPING\n"
                  "noise line\n"
                  "FIFI_QWEN36_MODEL_DIR is not set and has no entry in ~/.config/fifi/test-paths.json: SKIPPING\n")
        self.assertEqual(qi.collect_skips(stderr), {"FIFI_QWEN36_MODEL_DIR": 2})

    def test_plain_lines_are_not_skips(self) -> None:
        self.assertEqual(qi.collect_skips("error: something else: SKIPPING\n"), {})


class PolicyValidationTest(unittest.TestCase):
    def facts(self) -> dict:
        return {
            "targets": {
                "lib": {"path": "src/lib.rs", "test_fns": 1, "ignored": 0,
                        "env_gates": ["FLINT_QUAL_GO"],
                        "fixtures": {"Q36_CUDA_MODEL_DIR": {"or_skip": 0, "required": 1}}},
                "acq": {"path": "tests/acq.rs", "test_fns": 1, "ignored": 0,
                        "env_gates": [], "fixtures": {}},
            }
        }

    def policy(self, **overrides) -> dict:
        surface = {
            "id": "host-portable", "category": "host", "hardware": "cpu",
            "cwd": "engine", "argv": ["cargo", "test", "--locked"],
            "env": {}, "expect": {"min_ran": 1, "max_failed": 0},
            "fixtures_required": ["Q36_CUDA_MODEL_DIR"],
        }
        surface.update(overrides)
        return {"surfaces": [surface]}

    def test_valid_policy_resolves(self) -> None:
        ids = qi.validate_policy(self.policy(), self.facts())
        self.assertEqual(ids, ["host-portable"])

    def test_unknown_fixture_fails(self) -> None:
        with self.assertRaises(qi.InventoryError):
            qi.validate_policy(self.policy(fixtures_required=["NOPE"]), self.facts())

    def test_unknown_target_fails(self) -> None:
        with self.assertRaises(qi.InventoryError):
            qi.validate_policy(self.policy(targets_included=["nope"]), self.facts())

    def test_unknown_env_gate_fails(self) -> None:
        with self.assertRaises(qi.InventoryError):
            qi.validate_policy(self.policy(env={"NOT_A_GATE": "1"}), self.facts())

    def test_known_gate_via_env_passes(self) -> None:
        ids = qi.validate_policy(self.policy(env={"FLINT_QUAL_GO": "x"}), self.facts())
        self.assertEqual(ids, ["host-portable"])

    def test_bad_category_and_duplicate_ids_fail(self) -> None:
        with self.assertRaises(qi.InventoryError):
            qi.validate_policy(self.policy(category="quantum"), self.facts())
        bad = self.policy()
        bad["surfaces"].append(dict(bad["surfaces"][0]))
        with self.assertRaises(qi.InventoryError):
            qi.validate_policy(bad, self.facts())

    def test_allowed_failures_shape_is_checked(self) -> None:
        with self.assertRaises(qi.InventoryError):
            qi.validate_policy(
                self.policy(expect={"allowed_failures": ["no-separator"]}), self.facts()
            )


class AllowedFailureVerdictTest(unittest.TestCase):
    """Verdict logic of judge_surface over synthetic libtest output."""

    GOOD = "running 1 tests\ntest a ... ok\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.0s\n"
    BAD = "running 2 tests\ntest a ... ok\ntest b ... FAILED\ntest result: FAILED. 1 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.0s\n"
    STDERR = "     Running unittests src/lib.rs (x)\n"

    def judge(self, expect: dict, stdout: str, stderr: str, returncode: int,
              strict_fixtures: list[str] | None = None) -> tuple[str, list[str]]:
        verdict, reasons, _summary, _skips = qi.judge_surface(
            stdout, stderr, returncode, expect, strict_fixtures or []
        )
        return verdict, reasons

    def test_clean_run_passes(self) -> None:
        self.assertEqual(self.judge({}, self.GOOD, self.STDERR, 0), ("PASS", []))

    def test_unexpected_failure_fails(self) -> None:
        verdict, _ = self.judge({}, self.BAD, self.STDERR, 101)
        self.assertEqual(verdict, "FAIL")

    def test_allowed_failure_passes_with_raised_max_failed(self) -> None:
        expect = {"max_failed": 1, "allowed_failures": ["lib::b"]}
        self.assertEqual(self.judge(expect, self.BAD, self.STDERR, 101), ("PASS", []))

    def test_allowed_failure_that_stops_failing_fails(self) -> None:
        expect = {"max_failed": 1, "allowed_failures": ["lib::b"]}
        verdict, _ = self.judge(expect, self.GOOD, self.STDERR, 0)
        self.assertEqual(verdict, "FAIL")

    def test_min_ran_floor_fails_a_compile_error(self) -> None:
        verdict, _ = self.judge({"min_ran": 1}, "", "", 101)
        self.assertEqual(verdict, "FAIL")

    def test_strict_fixture_skip_fails(self) -> None:
        stderr = self.STDERR + (
            "Q36_CUDA_MODEL_DIR is not set and has no entry in "
            "~/.config/fifi/test-paths.json: SKIPPING\n"
        )
        verdict, _ = self.judge({}, self.GOOD, stderr, 0,
                                strict_fixtures=["Q36_CUDA_MODEL_DIR"])
        self.assertEqual(verdict, "FAIL")


class FactsDriftTest(unittest.TestCase):
    def test_validate_detects_stale_facts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            committed = {"schema_version": 1, "targets": {}}
            facts_path = Path(tmp) / "facts.json"
            facts_path.write_text(json.dumps(committed), encoding="utf-8")
            with mock.patch.object(qi, "FACTS_PATH", facts_path), \
                 mock.patch.object(qi, "POLICY_PATH", Path(tmp) / "missing.json"):
                # The stale committed facts are detected before the missing policy is read.
                with self.assertRaises(qi.InventoryError) as caught:
                    qi.cmd_validate(argparse.Namespace())
                self.assertIn("stale", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
