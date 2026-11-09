"""CPU-only checks for the P2P prefill measurement driver."""
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import bench_p2p_prefill as bench  # noqa: E402


def response(data):
    return io.BytesIO(json.dumps(data).encode())


class MeasurementDriverTest(unittest.TestCase):
    def test_self_test_still_runs_without_a_server(self):
        with mock.patch.object(sys, 'stdout', new_callable=io.StringIO) as out:
            bench.self_test()
        self.assertIn('measurement self-test passed', out.getvalue())

    def test_samples_are_recorded_with_a_reproducible_payload_and_median(self):
        info = {'engines': [{'model': 'fifi-q36', 'state': 'ready', 'context_limit': 131072}]}
        reply = {'id': 'sample', 'usage': {'prompt_tokens': 524, 'completion_tokens': 2},
                 'choices': [{'message': {'content': '4'}}]}
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'result.json'
            arguments = ['bench_p2p_prefill.py', '--samples', '2', '--output', str(out)]
            calls = [response(info), response(reply), response(info), response(reply)]
            with mock.patch.object(sys, 'argv', arguments), \
                 mock.patch.object(bench.urllib.request, 'urlopen', side_effect=calls), \
                 mock.patch.object(bench.time, 'monotonic', side_effect=[100, 110, 200, 210]):
                bench.main()
            data = json.loads(out.read_text())
            self.assertEqual(len(data['samples']), 2)
            self.assertEqual(data['median_seconds'], 10)
            self.assertEqual(data['median_prompt_tokens_per_second'], 52.4)
            self.assertEqual(data['samples'][0]['context_limit'], 131072)
            self.assertEqual(len(data['payload_sha256']), 64)
            self.assertFalse(out.with_name(out.name + '.partial').exists())

    def test_busy_engine_fails_without_starting_a_prompt(self):
        info = {'engines': [{'model': 'fifi-q36', 'state': 'busy', 'context_limit': 131072}]}
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'result.json'
            arguments = ['bench_p2p_prefill.py', '--samples', '1', '--output', str(out)]
            with mock.patch.object(sys, 'argv', arguments), \
                 mock.patch.object(bench.urllib.request, 'urlopen', return_value=response(info)) as http:
                with self.assertRaisesRegex(RuntimeError, 'not ready'):
                    bench.main()
            self.assertEqual(http.call_count, 1)
            self.assertFalse(out.exists())
            self.assertTrue(out.with_name(out.name + '.partial').exists())


class ReuseAccountingTest(unittest.TestCase):
    @staticmethod
    def reply(**extra_usage):
        return {
            'usage': {'prompt_tokens': 524, 'completion_tokens': 2, **extra_usage},
            'choices': [{'message': {'content': '4'}}],
        }

    def test_warm_rate_counts_only_new_prompt_tokens(self):
        for usage in ({'reused_tokens': 517}, {'prompt_tokens_details': {'cached_tokens': 517}},
                      {'reused_tokens': 517, 'prompt_tokens_details': {'cached_tokens': 517}}):
            with self.subTest(usage=usage):
                sample = bench.sample_of(self.reply(**usage), 0.2)
                self.assertEqual(sample['reuse_class'], 'warm')
                self.assertEqual(sample['reused_tokens'], 517)
                self.assertEqual(sample['new_prompt_tokens'], 7)
                self.assertEqual(sample['estimated_prompt_tokens_per_second'], 2620.0)
                self.assertEqual(sample['estimated_new_prompt_tokens_per_second'], 35.0)

    def test_cold_and_nested_usage_are_recorded(self):
        for usage in ({'reused_tokens': 0}, {'prompt_tokens_details': {'cached_tokens': 0}},
                      {'reused_tokens': 0, 'prompt_tokens_details': {'cached_tokens': 0}}):
            with self.subTest(usage=usage):
                sample = bench.sample_of(self.reply(**usage), 10)
                self.assertEqual(sample['reuse_class'], 'cold')
                self.assertEqual(sample['new_prompt_tokens'], 524)
                self.assertEqual(sample['estimated_new_prompt_tokens_per_second'], 52.4)

    def test_missing_reuse_is_unknown_not_cold(self):
        for usage in ({}, {'reused_tokens': None}, {'prompt_tokens_details': None},
                      {'prompt_tokens_details': {}}):
            with self.subTest(usage=usage):
                sample = bench.sample_of(self.reply(**usage), 10)
                self.assertEqual(sample['reuse_class'], 'unknown')
                self.assertIsNone(sample['reused_tokens'])
                self.assertIsNone(sample['new_prompt_tokens'])
                self.assertIsNone(sample['estimated_new_prompt_tokens_per_second'])

    def test_invalid_or_conflicting_reuse_is_rejected(self):
        bad = ({'reused_tokens': -1}, {'reused_tokens': True}, {'reused_tokens': '0'},
               {'reused_tokens': 525}, {'prompt_tokens_details': {'cached_tokens': 525}},
               {'prompt_tokens_details': {'cached_tokens': True}},
               {'prompt_tokens_details': 'bad'},
               {'reused_tokens': 0, 'prompt_tokens_details': {'cached_tokens': 1}})
        for usage in bad:
            with self.subTest(usage=usage):
                with self.assertRaises(ValueError):
                    bench.sample_of(self.reply(**usage), 1)

    def test_nonfinite_or_nonpositive_elapsed_is_rejected(self):
        for elapsed in (0, -1, True, '1', float('nan'), float('inf')):
            with self.subTest(elapsed=elapsed):
                with self.assertRaises(ValueError):
                    bench.sample_of(self.reply(reused_tokens=0), elapsed)

    def test_invalid_prompt_or_completion_counts_are_rejected(self):
        for field, value in (('prompt_tokens', True), ('prompt_tokens', 0),
                             ('completion_tokens', True), ('completion_tokens', 0),
                             ('completion_tokens', -1), ('completion_tokens', '2')):
            with self.subTest(field=field, value=value):
                reply = self.reply(reused_tokens=0)
                reply['usage'][field] = value
                with self.assertRaises(ValueError):
                    bench.sample_of(reply, 1)

    def test_mixed_samples_have_separate_cold_and_warm_summaries(self):
        info = {'engines': [{'model': 'fifi-q36', 'state': 'ready', 'context_limit': 131072}]}
        replies = [self.reply(reused_tokens=0), self.reply(reused_tokens=517), self.reply()]
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'result.json'
            arguments = ['bench_p2p_prefill.py', '--samples', '3', '--output', str(out)]
            calls = [response(item) for reply in replies for item in (info, reply)]
            with mock.patch.object(sys, 'argv', arguments), \
                 mock.patch.object(bench.urllib.request, 'urlopen', side_effect=calls), \
                 mock.patch.object(bench.time, 'monotonic', side_effect=[100, 104, 200, 200.2, 300, 302]):
                bench.main()
            data = json.loads(out.read_text())
            self.assertEqual(data['cold_samples'], 1)
            self.assertEqual(data['warm_samples'], 1)
            self.assertEqual(data['unknown_reuse_samples'], 1)
            self.assertEqual(data['median_cold_prompt_tokens_per_second'], 131.0)
            self.assertEqual(data['median_warm_new_prompt_tokens_per_second'], 35.0)
            self.assertEqual(data['median_prompt_tokens_per_second'], 262.0)


if __name__ == '__main__':
    unittest.main()
