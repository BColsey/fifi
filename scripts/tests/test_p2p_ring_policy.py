"""Check the native ring launch policy without a CUDA device."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
HEADER = ROOT / "engine/src/kernels/cuda/native"


class RingLaunchPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("g++-13") or shutil.which("g++")
        if compiler is None:
            raise RuntimeError("a C++ compiler is required for the ring policy test")
        cls.tmp = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        root = Path(cls.tmp.name)
        source = root / "policy.cc"
        cls.binary = root / "policy"
        source.write_text(
            '#include "peer_ring_policy.h"\n'
            "#include <cstdlib>\n"
            "#include <iostream>\n"
            "int main(int argc, char **argv) {\n"
            "  if (argc != 3) return 2;\n"
            "  const char *value = argv[1][0] == '-' ? nullptr : argv[1];\n"
            "  bool enabled = fifi_ring::multiblock_enabled(value);\n"
            "  std::cout << enabled << ' ' << fifi_ring::payload_blocks(std::atoi(argv[2]), enabled);\n"
            "}\n"
        )
        subprocess.run(
            [
                compiler,
                "-std=c++17",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-I",
                str(HEADER),
                str(source),
                "-o",
                str(cls.binary),
            ],
            check=True,
        )

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def policy(self, value, count):
        output = subprocess.check_output([self.binary, value, str(count)], text=True)
        enabled, blocks = map(int, output.split())
        return enabled, blocks

    def test_only_exact_one_enables_the_feature(self):
        for value in ("-", "", "0", "true", "01", "10", "1\n"):
            with self.subTest(value=value):
                self.assertEqual(self.policy(value, 524288), (0, 1))
        self.assertEqual(self.policy("1", 524288)[0], 1)

    def test_default_keeps_one_block_even_for_maximum_payload(self):
        for count in (1, 2048, 2049, 1 << 20):
            with self.subTest(count=count):
                self.assertEqual(self.policy("-", count), (0, 1))

    def test_opt_in_respects_the_single_block_boundary(self):
        self.assertEqual(self.policy("1", 2048), (1, 1))
        self.assertEqual(self.policy("1", 2049), (1, 2))

    def test_opt_in_caps_the_grid_at_64_blocks(self):
        self.assertEqual(self.policy("1", 131073), (1, 64))
        self.assertEqual(self.policy("1", 1 << 20), (1, 64))


if __name__ == "__main__":
    unittest.main()
