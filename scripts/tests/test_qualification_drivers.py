"""Runs the offline tests of the token-forcing device drivers with the scripts suite.

The drivers live in qualification/token-forcing/, which is internal: a published checkout
does not carry it, and there this module reports one skipped test. Elsewhere the scripts
suite (`python3 -I -m unittest discover -s scripts/tests`) runs the driver tests, so a
failing driver test fails it and every gate that runs it. Drivers without tests are an
error, not a skip, so deleting the tests cannot leave the suite green.
"""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

DRIVERS = Path(__file__).resolve().parents[2] / "qualification" / "token-forcing"


def driver_suite(drivers: Path) -> unittest.TestSuite | None:
    """The tests under `drivers/tests`, or None when the checkout has no `drivers` directory."""
    if not drivers.is_dir():
        return None
    tests = drivers / "tests"
    if not tests.is_dir():
        raise RuntimeError(f"{drivers} has no tests directory")
    # A fresh loader: discovery records its top-level directory on the loader that runs it.
    suite = unittest.TestLoader().discover(start_dir=str(tests), top_level_dir=str(tests))
    if suite.countTestCases() == 0:
        raise RuntimeError(f"{tests} holds no tests")
    return suite


def drivers_absent() -> None:
    raise unittest.SkipTest(f"{DRIVERS} is not in this checkout")


class DriverSuiteTests(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.drivers = Path(temp.name) / "token-forcing"

    def test_a_checkout_without_the_drivers_has_no_driver_suite(self) -> None:
        self.assertIsNone(driver_suite(self.drivers))

    def test_drivers_without_a_tests_directory_are_an_error(self) -> None:
        self.drivers.mkdir()
        (self.drivers / "qualify.py").write_text("VALUE = 1\n", encoding="utf-8")

        with self.assertRaisesRegex(RuntimeError, "has no tests directory"):
            driver_suite(self.drivers)

    def test_a_tests_directory_without_tests_is_an_error(self) -> None:
        (self.drivers / "tests").mkdir(parents=True)

        with self.assertRaisesRegex(RuntimeError, "holds no tests"):
            driver_suite(self.drivers)

    def test_the_drivers_tests_are_loaded(self) -> None:
        tests = self.drivers / "tests"
        tests.mkdir(parents=True)
        # A module name of its own, so discovery cannot meet an imported module of that name.
        (tests / "test_driver_suite_loader_probe.py").write_text(
            "import unittest\n\n\nclass Probe(unittest.TestCase):\n"
            "    def test_one(self):\n        pass\n\n    def test_two(self):\n        pass\n",
            encoding="utf-8",
        )

        # Discovery puts its top-level directory on sys.path; keep that inside this test.
        with mock.patch.object(sys, "path", list(sys.path)):
            suite = driver_suite(self.drivers)

        self.assertIsNotNone(suite)
        self.assertEqual(suite.countTestCases(), 2)


def load_tests(
    loader: unittest.TestLoader, standard_tests: unittest.TestSuite, pattern: str | None
) -> unittest.TestSuite:
    drivers = driver_suite(DRIVERS)
    if drivers is None:
        standard_tests.addTest(unittest.FunctionTestCase(drivers_absent))
    else:
        standard_tests.addTests(drivers)
    return standard_tests
