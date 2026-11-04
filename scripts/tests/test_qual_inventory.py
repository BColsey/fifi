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


class TornResultLineTest(unittest.TestCase):
    """Under --nocapture, test prints can tear libtest's result lines."""

    STDERR = "     Running unittests src/lib.rs (x)\n"

    def parse(self, body: str) -> dict:
        return parse_run(body, self.STDERR)

    def test_glued_status_still_counts(self) -> None:
        # Real shape from the integrated main: `test NAME ... ok` with the test's
        # REAL_ROW print glued onto the status word with no newline between.
        block = (
            "running 2 tests\n"
            "test one ... ok\n"
            "test pack::checks ... okREAL_ROW row=0 rel_l2=0.0 cosine=1.0\n"
            "test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; "
            "0 filtered out; finished in 0.0s\n"
        )
        self.assertEqual(self.parse(block)["totals"]["passed"], 2)

    def test_deferred_status_completed_by_bare_line(self) -> None:
        # The print lands between libtest's `test NAME ... ` prefix and its
        # status word, which then arrives alone on a later line.
        block = (
            "running 2 tests\n"
            "test one ... ok\n"
            "test quality::rows ... REAL_ROW row=1 rel_l2=1.0 cosine=0.0\n"
            "REAL_ROW row=0 rel_l2=inf cosine=0.0\n"
            "ok\n"
            "test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; "
            "0 filtered out; finished in 0.0s\n"
        )
        self.assertEqual(self.parse(block)["totals"]["passed"], 2)

    def test_parallel_pending_stack_completes_in_order(self) -> None:
        block = (
            "running 3 tests\n"
            "test a ... PRINT mid-run output\n"
            "test b ... ok\n"
            "test c ... PRINT other output\n"
            "ok\n"
            "ok\n"
            "test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; "
            "0 filtered out; finished in 0.0s\n"
        )
        self.assertEqual(self.parse(block)["totals"]["passed"], 3)

    def test_uncompleted_pending_still_fails_the_count(self) -> None:
        # A torn line whose status never arrives must not silently pass.
        block = (
            "running 2 tests\n"
            "test one ... ok\n"
            "test lost ... PRINT and never finishes\n"
            "test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; "
            "0 filtered out; finished in 0.0s\n"
        )
        with self.assertRaises(qi.InventoryError):
            self.parse(block)

    def test_phantom_full_result_line_still_overcounts(self) -> None:
        # Tolerance for torn lines must not accept fabricated full results: the
        # ghost line inflates the count past the summary's own arithmetic.
        block = (
            "running 2 tests\n"
            "test one ... ok\n"
            "test two ... ok\n"
            "test ghost ... ok\n"
            "test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; "
            "0 filtered out; finished in 0.0s\n"
        )
        with self.assertRaises(qi.InventoryError):
            self.parse(block)


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

    def test_phantom_running_line_from_test_output_fails_pairing(self) -> None:
        # Under --nocapture a test could print a line shaped like a block header;
        # the parser is start-anchored, so mid-line text is plain noise, but a
        # line-start phantom opens a block whose summary never arrives.
        noise_midline = "noise: running 1 tests\n" + OK_BLOCK
        stderr = "     Running unittests src/lib.rs (x)\n"
        summary = parse_run(noise_midline, stderr)
        self.assertEqual(summary["totals"]["passed"], 2)
        phantom_at_line_start = OK_BLOCK + "running 1 tests\n"
        with self.assertRaises(qi.InventoryError):
            parse_run(phantom_at_line_start, stderr)

    def test_inflated_test_line_count_fails_against_summary(self) -> None:
        # A stray line matching a test result inside an open block inflates the
        # counted lines past the summary's own arithmetic.
        poisoned = OK_BLOCK.replace(
            "test result:", "test ghost ... ok\ntest result:"
        )
        stderr = "     Running unittests src/lib.rs (x)\n"
        with self.assertRaises(qi.InventoryError):
            parse_run(poisoned, stderr)

    def test_duplicate_summary_line_is_noise_not_a_second_block(self) -> None:
        duplicated = OK_BLOCK + OK_BLOCK.splitlines()[-1] + "\n"
        stderr = "     Running unittests src/lib.rs (x)\n"
        summary = parse_run(duplicated, stderr)
        self.assertEqual(summary["totals"]["passed"], 2)

    def test_malformed_summary_is_noise_and_block_stays_open(self) -> None:
        malformed = (
            "running 1 tests\n"
            "test one ... ok\n"
            "test result: ok. nonsense passed; 0 failed\n"
        )
        with self.assertRaises(qi.InventoryError):
            qi.parse_stdout_blocks(malformed)


class SkipsTest(unittest.TestCase):
    def test_skipping_lines_collected_by_name(self) -> None:
        stderr = ("FIFI_QWEN36_MODEL_DIR is not set and has no entry in ~/.config/fifi/test-paths.json: SKIPPING\n"
                  "noise line\n"
                  "FIFI_QWEN36_MODEL_DIR is not set and has no entry in ~/.config/fifi/test-paths.json: SKIPPING\n")
        self.assertEqual(qi.collect_skips(stderr), {"FIFI_QWEN36_MODEL_DIR": 2})

    def test_plain_lines_are_not_skips(self) -> None:
        self.assertEqual(qi.collect_skips("error: something else: SKIPPING\n"), {})

    def test_integration_suite_skip_notice_counts(self) -> None:
        line = ("skipping: missing parity fixture /a/b (set FIFI_TEST_REQUIRE_DATA=1 "
                "to fail instead)\n")
        self.assertEqual(qi.collect_skips(line), {"": 1})


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

    def test_timeout_sec_must_be_positive(self) -> None:
        with self.assertRaises(qi.InventoryError):
            qi.validate_policy(self.policy(timeout_sec=0), self.facts())
        ids = qi.validate_policy(self.policy(timeout_sec=600), self.facts())
        self.assertEqual(ids, ["host-portable"])


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
        self.assertEqual(self.judge(expect, self.BAD, self.STDERR, 101),
                         ("PASS_WITH_KNOWN_FAILURE", []))

    def test_allowed_failure_that_stops_failing_fails(self) -> None:
        expect = {"max_failed": 1, "allowed_failures": ["lib::b"]}
        verdict, _ = self.judge(expect, self.GOOD, self.STDERR, 0)
        self.assertEqual(verdict, "FAIL")

    def test_min_ran_floor_fails_a_compile_error(self) -> None:
        verdict, _ = self.judge({"min_ran": 1}, "", "", 101)
        self.assertEqual(verdict, "FAIL")

    def test_fixture_skip_lines_deflate_executed_effective(self) -> None:
        stderr = self.STDERR + (
            "Q36_CUDA_MODEL_DIR is not set and has no entry in "
            "~/.config/fifi/test-paths.json: SKIPPING\n"
        )
        verdict, reasons, summary, _skips = qi.judge_surface(
            self.GOOD, stderr, 0, {"min_ran": 1}, []
        )
        assert summary is not None
        self.assertEqual(summary["accounting"]["executed_effective"], 0)
        self.assertEqual(verdict, "FAIL")
        self.assertTrue(any("executed 0" in r for r in reasons))

    def test_ignored_tests_never_count_toward_the_floor(self) -> None:
        # The review case: 2988 passed, 1 failed, 36 ignored, 1 skip line.
        # executed_effective must be 2988, not 3024: ignored tests did not run.
        ignored = [f"test i{n} ... ignored" for n in range(36)]
        lines = ["running 3025 tests"]
        lines += ["test p ... ok"] * 2988
        lines += ignored
        lines += ["test baseline ... FAILED"]
        lines += ["test result: FAILED. 2988 passed; 1 failed; 36 ignored; "
                  "0 measured; 0 filtered out; finished in 1.0s"]
        stderr = self.STDERR + (
            "skipping: missing parity fixture /x/y (set FIFI_TEST_REQUIRE_DATA=1 "
            "to fail instead)\n"
        )
        verdict, _reasons, summary, _skips = qi.judge_surface(
            "\n".join(lines) + "\n", stderr, 101,
            {"min_ran": 3000, "max_failed": 1,
             "allowed_failures": ["lib::baseline"]}, [],
        )
        assert summary is not None
        acc = summary["accounting"]
        self.assertEqual(acc["executed_effective"], 2988)
        self.assertEqual(acc["discovered"], 3025)
        self.assertEqual(acc["ignored"], 36)
        self.assertEqual(acc["fixture_skipped"], 1)
        self.assertEqual(verdict, "FAIL")  # 2988 < 3000: the floor must bite

    def test_floor_met_with_ignored_excluded(self) -> None:
        ignored_block = "test i0 ... ignored, device only\n"
        block = (
            "running 3 tests\n"
            "test a ... ok\n"
            "test b ... ok\n"
            + ignored_block +
            "test result: ok. 2 passed; 0 failed; 1 ignored; 0 measured; "
            "0 filtered out; finished in 0.0s\n"
        )
        verdict, _reasons, summary, _skips = qi.judge_surface(
            block, self.STDERR, 0, {"min_ran": 2}, []
        )
        assert summary is not None
        self.assertEqual(summary["accounting"]["executed_effective"], 2)
        self.assertEqual(verdict, "PASS")

    def test_allowed_failure_reports_known_failure_verdict(self) -> None:
        expect = {"max_failed": 1, "allowed_failures": ["lib::b"]}
        verdict, _ = self.judge(expect, self.BAD, self.STDERR, 101)
        self.assertEqual(verdict, "PASS_WITH_KNOWN_FAILURE")

    def test_clean_run_stays_plain_pass(self) -> None:
        expect = {"max_failed": 1, "allowed_failures": ["lib::b"]}
        # The waiver exists but nothing failed; require-all already failed the
        # verdict, so this cannot be misread as a clean PASS either.
        verdict, _ = self.judge(expect, self.GOOD, self.STDERR, 0)
        self.assertNotEqual(verdict, "PASS")

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
