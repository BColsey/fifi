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


if __name__ == '__main__':
    unittest.main()
