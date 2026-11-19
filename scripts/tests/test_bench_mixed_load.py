from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bench_mixed_load as mixed


class MixedLoadTests(unittest.TestCase):
    def pair(self, first, admission):
        fixture = {"cold": "cold", "count": "count"}

        def simulate(client, model, name, notify):
            if name == first:
                notify("admitted")
                notify("visible")
                return dict(start=0, admitted=0.1, times=[2, 4], end=5)
            return dict(start=admission - 0.01, admitted=admission, times=[6, 7], end=8)

        with patch.object(mixed, "timed_stream", simulate):
            return mixed.run_pair(None, "model", fixture, first)

    def test_second_admitted_during_decode_or_prefill(self):
        for first, admission in [("count", 3), ("cold", 1)]:
            with self.subTest(first=first):
                self.assertEqual(set(self.pair(first, admission)), {"count", "cold"})

    def test_late_admission_rejects_false_overlap(self):
        for first, admission in [("count", 4), ("cold", 2)]:
            with self.subTest(first=first), self.assertRaisesRegex(ValueError, "admitted"):
                self.pair(first, admission)

    def test_false_warm_usage(self):
        with self.assertRaises(ValueError):
            mixed.checked_reuse({"prompt_tokens": 300, "reused_tokens": 0,
                                 "prompt_tokens_details": {"cached_tokens": 299}})

    def test_control_requires_an_ordered_count_and_length_stop(self):
        sample = {"pieces": ["\n".join(map(str, range(1, 40)))],
                  "stream": {"usage": {"completion_tokens": 128}, "finish_reason": "length"}}
        self.assertTrue(mixed.valid_count(sample))
        sample["pieces"].append("\n41")
        self.assertFalse(mixed.valid_count(sample))

    def test_percentile_floor_and_nearest_rank(self):
        self.assertIsNone(mixed.p95(list(range(19))))
        self.assertEqual(mixed.p95(list(range(1, 21))), 19)
