"""Offline tests for the qualification inventory script: census, parsing, and validation.

The collector's process running (cargo, git) is exercised by real runs recorded in
the task evidence; these tests pin the pure logic that makes evidence trustworthy.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time
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

# Byte-exact label shape cargo emits under CARGO_TERM_COLOR=always
# (captured from a real run; bold + bright-green label, reset after).
COLORED_CARGO_STDERR = (
    "\x1b[0m\x1b[1m\x1b[92m   Compiling\x1b[0m inference-engine v0.1.0\n"
    "\x1b[1m\x1b[92m     Running\x1b[0m unittests src/lib.rs "
    "(target/debug/deps/inference_engine-abc)\n"
    "\x1b[1m\x1b[92m     Running\x1b[0m tests/acquisition.rs "
    "(target/debug/deps/acquisition-def)\n"
    "\x1b[1m\x1b[92m   Doc-tests\x1b[0m inference_engine\n"
)


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


class CapturedTornOutputTest(unittest.TestCase):
    """Permanent regression fixtures: verbatim fragments of the two real
    integration runs whose torn result lines failed the collection
    (run3/run4 at the slice-1 integration; REAL_ROW and runner prints from
    the tests landed inside libtest's no-newline result prefixes)."""

    FIXTURES = Path(__file__).parent / "fixtures"
    STDERR = "     Running unittests src/lib.rs (x)\n"

    def test_glued_and_deferred_shapes_from_run3(self) -> None:
        fragment = (self.FIXTURES / "torn_glue_defer.txt").read_text(encoding="utf-8")
        names = {
            line.split(" ... ")[0][len("test "):]
            for line in fragment.splitlines()
            if line.startswith("test ") and " ... " in line
        }
        block = f"running {len(names)} tests\n{fragment}" \
                f"test result: ok. {len(names)} passed; 0 failed; 0 ignored; " \
                "0 measured; 0 filtered out; finished in 0.0s\n"
        summary = qi.summarize_run(block, self.STDERR)
        counted = [name for name, _status in
                   qi.parse_stdout_blocks(block)[0].lines]
        self.assertEqual(len(counted), len(names))
        self.assertEqual(set(counted), names)  # glued and deferred all recovered
        self.assertEqual(summary["totals"]["passed"], len(names))

    def test_concurrent_deferred_completions_from_run4(self) -> None:
        fragment = (self.FIXTURES / "torn_concurrent.txt").read_text(encoding="utf-8")
        summary = qi.summarize_run(fragment, self.STDERR)
        self.assertEqual(summary["totals"]["passed"], 2)  # both bare `ok` lines landed


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

#![cfg(all(feature = "native-cuda", target_os = "linux"))]
#[cfg(feature = "native-mlx")]
fn gated() {}
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.rs"
            path.write_text(source, encoding="utf-8")
            fns, ignored, gates, features, fixtures = qi.scan_file(path)
        self.assertEqual(fns, 3)
        self.assertEqual(ignored, 1)
        self.assertEqual(gates, {"FLINT_QUAL_GO", "FIFI_FLINT_GGUF"})
        self.assertEqual(features, {"native-cuda", "native-mlx"})
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

    def test_skipping_lines_survive_cargo_term_color_always(self) -> None:
        line = ("\x1b[1m\x1b[32mFIFI_QWEN36_MODEL_DIR\x1b[0m is not set and has "
                "no entry in ~/.config/fifi/test-paths.json: SKIPPING\n")
        self.assertEqual(qi.collect_skips(line), {"FIFI_QWEN36_MODEL_DIR": 1})


class ColoredCargoStderrTest(unittest.TestCase):
    """CI forces CARGO_TERM_COLOR=always; the GitHub-runner collector saw
    72 stdout blocks and zero target headers until the escapes were
    stripped at the parser boundary."""

    def test_running_headers_parse_under_color(self) -> None:
        self.assertEqual(qi.parse_cargo_targets(COLORED_CARGO_STDERR),
                         ["lib", "acquisition", "doctest"])

    def test_plain_stderr_unchanged(self) -> None:
        self.assertEqual(qi.parse_cargo_targets(CARGO_STDERR),
                         ["lib", "acquisition", "doctest"])

    def test_strip_ansi_leaves_plain_text_untouched(self) -> None:
        self.assertEqual(qi.ANSI_ESCAPE_RE.sub("", CARGO_STDERR), CARGO_STDERR)


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
              strict_fixtures: list[str] | None = None,
              expected: list[str] | None = None) -> tuple[str, list[str]]:
        verdict, reasons, _summary, _skips = qi.judge_surface(
            stdout, stderr, returncode, expect, strict_fixtures or [],
            expected or ["lib"],
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
            self.GOOD, stderr, 0, {"min_ran": 1}, [], ["lib"]
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
             "allowed_failures": ["lib::baseline"]}, [], ["lib"],
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
            block, self.STDERR, 0, {"min_ran": 2}, [], ["lib"]
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


class DeviceSurfacePolicyTest(unittest.TestCase):
    """Slice 2: device surfaces are declarable, but only fail-closed."""

    def facts(self) -> dict:
        return {
            "cargo_features": ["native-cuda", "native-mlx"],
            "targets": {
                "lib": {"path": "src/lib.rs", "test_fns": 1, "ignored": 0,
                        "env_gates": ["INFERENCE_CUDA_REQUIRE_DEVICE"],
                        "features": [], "fixtures": {}},
                "cuda_smoke": {"path": "tests/cuda_smoke.rs", "test_fns": 4,
                               "ignored": 0, "env_gates": ["INFERENCE_CUDA_REQUIRE_DEVICE"],
                               "features": ["native-cuda"], "fixtures": {}},
                "native_fast": {"path": "tests/native_fast.rs", "test_fns": 6,
                                "ignored": 0, "env_gates": [],
                                "features": ["native-mlx"], "fixtures": {}},
            },
        }

    def cuda_surface(self, **overrides) -> dict:
        surface = {
            "id": "cuda-smoke", "category": "cuda-device",
            "hardware": "nvidia-gpu",
            "description": "seed device surface",
            "cwd": "engine",
            "argv": ["cargo", "test", "--locked", "--features", "native-cuda",
                     "--test=cuda_smoke"],
            "features": ["native-cuda"],
            "env": {"INFERENCE_CUDA_REQUIRE_DEVICE": "1"},
            "expect": {"min_ran": 4},
            "timeout_sec": 900,
        }
        surface.update(overrides)
        return surface

    def check(self, surface: dict) -> None:
        qi.validate_policy({"surfaces": [surface]}, self.facts())

    def test_fail_closed_cuda_surface_resolves(self) -> None:
        self.check(self.cuda_surface())

    def test_cuda_surface_without_require_device_fails(self) -> None:
        with self.assertRaises(qi.InventoryError) as caught:
            self.check(self.cuda_surface(env={}))
        self.assertIn("INFERENCE_CUDA_REQUIRE_DEVICE", str(caught.exception))

    def test_cuda_surface_with_zero_require_device_fails(self) -> None:
        with self.assertRaises(qi.InventoryError):
            self.check(self.cuda_surface(env={"INFERENCE_CUDA_REQUIRE_DEVICE": "0"}))

    def test_cuda_surface_without_min_ran_fails(self) -> None:
        # A compiled-but-empty target (runtime gates, ignored tests) must fail on
        # its own; feature validation alone cannot catch zero executed tests.
        with self.assertRaises(qi.InventoryError) as caught:
            self.check(self.cuda_surface(expect={}))
        self.assertIn("min_ran", str(caught.exception))

    def test_cuda_surface_with_zero_min_ran_fails(self) -> None:
        with self.assertRaises(qi.InventoryError):
            self.check(self.cuda_surface(expect={"min_ran": 0}))

    def test_undeclared_feature_fails(self) -> None:
        with self.assertRaises(qi.InventoryError):
            self.check(self.cuda_surface(features=["native-cuda", "not-a-feature"]))

    def test_argv_missing_target_feature_fails(self) -> None:
        # cuda_smoke's source gates on native-cuda; a surface that runs it without
        # the feature compiles the target empty and would report zero tests.
        # (two-argument --test form)
        with self.assertRaises(qi.InventoryError) as caught:
            self.check(self.cuda_surface(features=["native-mlx"],
                                         argv=["cargo", "test", "--locked",
                                               "--features", "native-mlx",
                                               "--test", "cuda_smoke"]))
        self.assertIn("gates on", str(caught.exception))

    def test_facts_feature_unknown_to_cargo_fails(self) -> None:
        bad = self.facts()
        bad["targets"]["cuda_smoke"]["features"] = ["ghost-feature"]
        with self.assertRaises(qi.InventoryError):
            qi.validate_policy({"surfaces": [self.cuda_surface()]}, bad)


class MultilineFeatureGateTest(unittest.TestCase):
    """A formatting change must never silently weaken the gate census."""

    def scan(self, source: str) -> set[str]:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.rs"
            path.write_text(source, encoding="utf-8")
            return qi.scan_file(path)[3]

    def test_multiline_cfg_attribute_is_counted(self) -> None:
        source = (
            "#![cfg(all(\n"
            "    feature = \"native-cuda\",\n"
            "    target_os = \"linux\",\n"
            "))]\n"
            "#[cfg(all(feature = \"native-mlx\",\n"
            "           target_os = \"macos\"))]\n"
            "fn gated() {}\n"
        )
        self.assertEqual(self.scan(source), {"native-cuda", "native-mlx"})

    def test_cfg_macro_features_are_not_target_gates(self) -> None:
        # The probe.rs shape: cfg!(...) evaluates at runtime and never gates
        # target compilation, so it neither counts nor refuses.
        source = (
            "Self {\n"
            "    cuda: cfg!(all(feature = \"native-cuda\", target_os = \"linux\")),\n"
            "    mlx: cfg!(all(\n"
            "        feature = \"native-mlx\",\n"
            "        target_os = \"macos\",\n"
            "        target_arch = \"aarch64\"\n"
            "    )),\n"
            "}\n"
        )
        self.assertEqual(self.scan(source), set())

    def test_bare_feature_line_refuses_instead_of_undercounting(self) -> None:
        # A raw string or odd formatting can put `feature = "x"` outside any
        # recognizable attribute; refusing beats guessing a smaller census.
        source = 'let note = r#"feature = "native-cuda""#;\n'
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.rs"
            path.write_text(source, encoding="utf-8")
            with self.assertRaises(qi.InventoryError) as caught:
                qi.scan_file(path)
        self.assertIn("refusing", str(caught.exception))


class SurfaceSelectionTest(unittest.TestCase):
    def policy(self) -> dict:
        cpu = {"id": "host-portable", "category": "host", "hardware": "cpu"}
        gpu = {"id": "cuda-smoke", "category": "cuda-device", "hardware": "nvidia-gpu"}
        return {"surfaces": [cpu, gpu]}

    def test_default_selection_is_cpu_only(self) -> None:
        # A leaseless `collect` must never launch GPU work by accident.
        selected, skipped = qi.select_surfaces(self.policy(), None)
        self.assertEqual(selected, ["host-portable"])
        self.assertEqual(skipped, ["cuda-smoke"])

    def test_explicit_selection_reaches_device_surfaces(self) -> None:
        selected, skipped = qi.select_surfaces(self.policy(), ["cuda-smoke"])
        self.assertEqual(selected, ["cuda-smoke"])
        self.assertEqual(skipped, ["host-portable"])


class ProcessGroupTimeoutTest(unittest.TestCase):
    def test_timeout_kills_the_whole_process_group(self) -> None:
        # The reviewer's rule: a timed-out run must not orphan the test binaries
        # (or GPU work) that cargo spawned. The child spawns a grandchild sleeper
        # in the same process group; the kill must take both.
        child = (
            "import subprocess, sys, time\n"
            "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
            "print(p.pid, flush=True)\n"
            "time.sleep(30)\n"
        )
        surface = {
            "id": "grp", "category": "host", "hardware": "cpu",
            "cwd": "scripts", "argv": [sys.executable, "-c", child],
            "env": {}, "timeout_sec": 1,
        }
        with tempfile.TemporaryDirectory() as tmp:
            record = qi.collect_surface(surface, Path(tmp), {"targets": {}})
            grandchild = int((Path(tmp) / "grp.stdout").read_text().strip())
        self.assertEqual(record["verdict"], "FAIL")
        self.assertTrue(any("timed out" in r for r in record["reasons"]))
        for _ in range(50):
            try:
                os.kill(grandchild, 0)
            except ProcessLookupError:
                return  # the grandchild died with the group
            time.sleep(0.1)
        self.fail(f"grandchild {grandchild} survived the timeout kill")


class PackRegistryTest(unittest.TestCase):
    """The pack registry binds to loader pins and the preflight mapping."""

    @staticmethod
    def _registry() -> dict:
        return json.loads((ROOT / "scripts" / "pack_registry.json").read_text())

    def setUp(self):
        self._patches = (
            mock.patch.object(qi, "gguf_pin_name", return_value="Qwen3.6-35B-A3B-Q4_K_M.gguf"),
            mock.patch.object(qi, "tokenizer_pin_sha256s", return_value={
                "tokenizer_json": "a" * 64,
                "tokenizer_config_json": "b" * 64,
                "chat_template_jinja": "c" * 64,
                "generation_config_json": "d" * 64,
            }),
            mock.patch.object(qi, "weight_format_qualification",
                              return_value={"gguf": "qualified", "exl3": "experimental"}),
        )
        for patch in self._patches:
            patch.start()
            self.addCleanup(patch.stop)

    def _bound(self) -> dict:
        """A registry whose gguf identity matches the patched source pins."""
        registry = self._registry()
        registry["packs"][0]["identity"]["tokenizer_pin_sha256"] = {
            "tokenizer_json": "a" * 64,
            "tokenizer_config_json": "b" * 64,
            "chat_template_jinja": "c" * 64,
            "generation_config_json": "d" * 64,
        }
        return registry

    def test_committed_registry_validates_against_real_source(self):
        # The real extractor functions run here (setUp patches only the
        # fixtures below): the committed registry must bind to the loader
        # constants and the preflight mapping.
        for patch in self._patches:
            patch.stop()
        self.addCleanup(lambda: [p.start() for p in self._patches])
        self.assertEqual(qi.cmd_packs(argparse.Namespace()), 0)

    def test_bound_registry_resolves(self):
        self.assertEqual(
            qi.validate_pack_registry(self._bound()),
            {"qwen36/gguf": "reference", "qwen36/exl3-4.00": "fail", "qwen36/exl3-4.50": "fail"},
        )

    def test_tokenizer_pin_drift_fails(self):
        registry = self._bound()
        registry["packs"][0]["identity"]["tokenizer_pin_sha256"]["tokenizer_json"] = "f" * 64
        with self.assertRaisesRegex(qi.InventoryError, "tokenizer pin tokenizer_json drifted"):
            qi.validate_pack_registry(registry)

    def test_pinned_name_drift_fails(self):
        registry = self._bound()
        registry["packs"][0]["identity"]["pinned_gguf_name"] = "other.gguf"
        with self.assertRaisesRegex(qi.InventoryError, "!= loader pin"):
            qi.validate_pack_registry(registry)

    def test_runtime_label_contradicting_preflight_fails(self):
        registry = self._bound()
        registry["packs"][0]["production_eligibility"]["runtime_label"] = "experimental"
        with self.assertRaisesRegex(qi.InventoryError, "contradicts the preflight mapping"):
            qi.validate_pack_registry(registry)

    def test_missing_numerical_verdict_fails(self):
        registry = self._bound()
        del registry["packs"][1]["numerical_qualification"]
        with self.assertRaisesRegex(qi.InventoryError, "numerical_qualification must have"):
            qi.validate_pack_registry(registry)

    def test_no_production_default_fails(self):
        registry = self._bound()
        for pack in registry["packs"]:
            pack["production_eligibility"]["production_default"] = False
        with self.assertRaisesRegex(qi.InventoryError, "exactly one production_default"):
            qi.validate_pack_registry(registry)

    def test_default_must_be_qualified_gguf(self):
        registry = self._bound()
        registry["packs"][0]["production_eligibility"]["production_default"] = False
        registry["packs"][1]["production_eligibility"]["production_default"] = True
        with self.assertRaisesRegex(qi.InventoryError, "qualified gguf pack"):
            qi.validate_pack_registry(registry)

    def test_reference_verdict_only_for_default(self):
        registry = self._bound()
        registry["packs"][2]["numerical_qualification"]["verdict"] = "reference"
        with self.assertRaisesRegex(qi.InventoryError, "only the production default may be the reference"):
            qi.validate_pack_registry(registry)

    def test_failed_source_binding_is_not_a_silent_pass(self):
        with mock.patch.object(qi, "gguf_pin_name",
                               side_effect=qi.InventoryError("source gone")):
            with self.assertRaisesRegex(qi.InventoryError, "source binding failed"):
                qi.validate_pack_registry(self._bound())

    def test_filename_cannot_be_reported_as_cryptographic(self):
        registry = self._bound()
        registry["identity_contract"]["fields"]["gguf_expected_filename"] = "cryptographically pinned"
        with self.assertRaisesRegex(qi.InventoryError, "inflating identity assurance"):
            qi.validate_pack_registry(registry)

    def test_tensor_schema_cannot_be_reported_as_cryptographic(self):
        registry = self._bound()
        registry["identity_contract"]["fields"]["exl3_tensor_schema"] = "cryptographically pinned"
        with self.assertRaisesRegex(qi.InventoryError, "exl3_tensor_schema must read"):
            qi.validate_pack_registry(registry)

    def test_exact_pack_contents_must_stay_unestablished(self):
        registry = self._bound()
        registry["identity_contract"]["fields"]["exl3_exact_pack_contents"] = "shape-validated"
        with self.assertRaisesRegex(qi.InventoryError, "exl3_exact_pack_contents must read"):
            qi.validate_pack_registry(registry)

    def test_pack_cannot_claim_weight_digests(self):
        # A pack asserting a weights sha256 would claim an identity the
        # runtime never verifies; strict identity keys must refuse it.
        registry = self._bound()
        registry["packs"][2]["identity"]["weights_sha256"] = "f" * 64
        with self.assertRaisesRegex(qi.InventoryError, "exl3 identity needs a non-empty bpw"):
            qi.validate_pack_registry(registry)

    def test_scorer_margins_are_read_from_the_compiled_source(self):
        compiled = qi.scorer_margins()
        self.assertEqual(compiled["v6"], (0.02, -0.005, -0.01, -0.02))
        self.assertEqual(compiled["v7"], (0.02, -0.005, 0.02, 0.02))

    def test_registration_margin_drift_from_source_fails(self):
        registry = self._bound()
        registry["quality_standard"]["registrations"]["versions"]["v7"]["gates"]["copy"]["margin"] = 0.03
        with self.assertRaisesRegex(qi.InventoryError, "declared margins"):
            qi.validate_pack_registry(registry)

    def test_v6_registration_stays_frozen_against_source(self):
        registry = self._bound()
        registry["quality_standard"]["registrations"]["versions"]["v6"]["gates"]["retrieval"]["margin"] = -0.07
        with self.assertRaisesRegex(qi.InventoryError, "declared margins"):
            qi.validate_pack_registry(registry)

    def test_v7_cannot_change_the_nll_or_top1_gates(self):
        registry = self._bound()
        v7 = registry["quality_standard"]["registrations"]["versions"]["v7"]
        v7["gates"]["nll"]["statistic"] = "something else"
        with self.assertRaisesRegex(qi.InventoryError, "v7 nll gate must stay identical"):
            qi.validate_pack_registry(registry)

    def test_v7_must_change_the_probe_statistics(self):
        registry = self._bound()
        versions = registry["quality_standard"]["registrations"]["versions"]
        versions["v7"]["gates"]["copy"]["statistic"] = versions["v6"]["gates"]["copy"]["statistic"]
        with self.assertRaisesRegex(qi.InventoryError, "v7 must change the copy probe statistic"):
            qi.validate_pack_registry(registry)

    def test_unknown_registration_version_fails_closed(self):
        registry = self._bound()
        registry["quality_standard"]["registrations"]["versions"]["v8"] = {}
        with self.assertRaisesRegex(qi.InventoryError, "versions must be exactly"):
            qi.validate_pack_registry(registry)
        registry = self._bound()
        registry["quality_standard"]["registrations"]["current"] = "v8"
        with self.assertRaisesRegex(qi.InventoryError, "current must name a declared version"):
            qi.validate_pack_registry(registry)
        registry = self._bound()
        registry["packs"][1]["numerical_qualification"]["registered_under"] = "v5"
        with self.assertRaisesRegex(qi.InventoryError, "registered_under must name a declared"):
            qi.validate_pack_registry(registry)

    def test_registered_under_needs_its_annotation(self):
        registry = self._bound()
        del registry["packs"][1]["numerical_qualification"]["annotation"]
        with self.assertRaisesRegex(qi.InventoryError, "come together"):
            qi.validate_pack_registry(registry)

    def test_bootstrap_seed_and_draws_cannot_drift(self):
        registry = self._bound()
        registry["quality_standard"]["registrations"]["versions"]["v7"]["bootstrap"]["seed"] = 1
        with self.assertRaisesRegex(qi.InventoryError, "frozen seed 20261031"):
            qi.validate_pack_registry(registry)

    def test_existing_exl3_verdicts_are_annotated_not_rejudged(self):
        for pack in self._registry()["packs"][1:]:
            numerical = pack["numerical_qualification"]
            self.assertEqual(numerical["verdict"], "fail")
            self.assertEqual(numerical["registered_under"], "v6")
            self.assertIn("not re-judged", numerical["annotation"])

    def test_qualified_label_cannot_exceed_failed_evidence(self):
        registry = self._bound()
        registry["packs"][1]["production_eligibility"]["runtime_label"] = "qualified"
        # The preflight mapping check fires first (exl3 maps to experimental
        # in source); neutralize it so the evidence rule is what fails.
        with mock.patch.object(qi, "weight_format_qualification",
                               return_value={"gguf": "qualified", "exl3": "qualified"}):
            with self.assertRaisesRegex(qi.InventoryError, "exceeds its recorded"):
                qi.validate_pack_registry(registry)


class TargetCoverageTest(unittest.TestCase):
    """The completeness contract: a verdict must account for every target.

    Regression suite for the fail-fast blind spot: the slice-3 host receipt
    reported PASS_WITH_KNOWN_FAILURE while cargo's default fail-fast had
    aborted nine-plus test executables that never ran. The floor alone must
    never imply coverage.
    """

    OK = "running 1 tests\ntest a ... ok\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.0s\n"

    @staticmethod
    def stderr(*names: str) -> str:
        lines = ["     Compiling inference-engine v0.1.0"]
        for name in names:
            if name == "lib":
                lines.append("     Running unittests src/lib.rs (x)")
            elif name == "doctest":
                lines.append("     Doc-tests inference_engine")
            else:
                lines.append(f"     Running tests/{name}.rs (x)")
        return "\n".join(lines) + "\n"

    def judge(self, expected: list[str], observed: list[str]):
        stdout = "".join(self.OK for _ in observed)
        return qi.judge_surface(
            stdout, self.stderr(*observed), 0,
            {"min_ran": 1}, [], expected,
        )

    def test_missing_target_refuses_even_above_the_floor(self) -> None:
        # A suppressed executable: expected three targets, cargo ran two.
        verdict, reasons, _summary, _skips = self.judge(
            ["lib", "published_text", "doctest"], ["lib", "published_text"]
        )
        self.assertEqual(verdict, "FAIL")
        self.assertTrue(any("never ran: doctest" in r for r in reasons))

    def test_fail_fast_abort_shape_is_the_slice3_regression(self) -> None:
        # Exactly the historical defect: everything through the failing binary
        # ran, the tail never launched, and the floor was still satisfied.
        expected = ["lib", "published_text", "q36_real_config", "quantization",
                    "server_q36_cli", "tiered_weights", "doctest"]
        observed = ["lib", "published_text", "q36_real_config"]
        verdict, reasons, summary, _skips = self.judge(expected, observed)
        self.assertEqual(verdict, "FAIL")
        self.assertIn("incomplete target coverage — cargo never ran: "
                      "quantization, server_q36_cli, tiered_weights, doctest",
                      reasons)
        assert summary is not None
        self.assertEqual(summary["target_coverage"]["missing"],
                         ["quantization", "server_q36_cli", "tiered_weights",
                          "doctest"])

    def test_truncated_stream_fails_instead_of_pairing_survivors(self) -> None:
        # Truncated stdout: a Running line lost its test block. The block/target
        # pairing must refuse before any verdict can pass.
        with self.assertRaises(qi.InventoryError):
            qi.summarize_run(self.OK, self.stderr("lib", "acquisition"))

    def test_duplicated_target_block_fails(self) -> None:
        # A duplicated result: the same executable reported twice.
        verdict, reasons, _summary, _skips = self.judge(
            ["lib", "acquisition"], ["lib", "lib", "acquisition"]
        )
        self.assertEqual(verdict, "FAIL")
        self.assertTrue(any("duplicate target blocks: lib" in r for r in reasons))

    def test_unknown_observed_target_fails(self) -> None:
        verdict, reasons, _summary, _skips = self.judge(
            ["lib"], ["lib", "mystery"]
        )
        self.assertEqual(verdict, "FAIL")
        self.assertTrue(any("facts do not declare: mystery" in r for r in reasons))

    def test_complete_coverage_passes_and_is_recorded(self) -> None:
        verdict, reasons, summary, _skips = self.judge(
            ["lib", "acquisition", "doctest"], ["lib", "acquisition", "doctest"]
        )
        self.assertEqual((verdict, reasons), ("PASS", []))
        assert summary is not None
        self.assertEqual(summary["target_coverage"],
                         {"expected": 3, "observed": 3, "missing": [],
                          "duplicates": [], "unknown": []})

    def test_live_inventory_names_bins_and_doctests(self) -> None:
        # The facts generator must derive the inventory from the tree, not a
        # hardcoded list: bin unittests and doc-tests included.
        names = qi.live_target_names()
        for required in ("lib", "main", "doctest", "published_text",
                         "server_q36_cli"):
            self.assertIn(required, names)


if __name__ == "__main__":
    unittest.main()
