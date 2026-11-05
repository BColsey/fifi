"""Runs the offline tests of the token-forcing device drivers with the scripts suite.

The drivers live in qualification/token-forcing/, which is internal: a published checkout
does not carry it, and there this module reports one skipped test. Elsewhere the scripts
suite (`python3 -I -m unittest discover -s scripts/tests`) runs the driver tests, so a
failing driver test fails it and every gate that runs it.
"""

from __future__ import annotations

from pathlib import Path
import unittest

DRIVER_TESTS = Path(__file__).resolve().parents[2] / "qualification" / "token-forcing" / "tests"


class DriversAbsent(unittest.TestCase):
    def test_driver_tests(self) -> None:
        self.skipTest(f"{DRIVER_TESTS} is not in this checkout")


def load_tests(
    loader: unittest.TestLoader, standard_tests: unittest.TestSuite, pattern: str | None
) -> unittest.TestSuite:
    if not DRIVER_TESTS.is_dir():
        return loader.loadTestsFromTestCase(DriversAbsent)
    # A fresh loader: discovery records its top-level directory on the loader that runs it.
    drivers = unittest.TestLoader().discover(
        start_dir=str(DRIVER_TESTS), top_level_dir=str(DRIVER_TESTS)
    )
    if drivers.countTestCases() == 0:
        raise RuntimeError(f"{DRIVER_TESTS} holds no tests")
    return drivers
