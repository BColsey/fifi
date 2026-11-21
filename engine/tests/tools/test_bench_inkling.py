import copy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import bench_inkling as bench


class BenchmarkTests(unittest.TestCase):
    def test_readiness_accepts_only_the_bound_loopback_worker(self):
        args = SimpleNamespace(startup_timeout=1, request_timeout=1, context=128)
        process = Mock()
        process.poll.return_value = None
        with TemporaryDirectory() as path, patch.object(bench, "Server") as server:
            log = Path(path) / "server.log"
            log.write_text("enginesrv listening on 127.0.0.1:51234\n")
            client = server.return_value
            client.json.return_value = {
                "engines": [{"id": "inkling", "model": "inkling", "state": "ready", "context_limit": 128}],
                "backends_compiled": ["mlx"],
            }
            report = {}
            self.assertIs(bench.wait_until_ready(process, log, "test-key", args, report,
                                                  bench.time.monotonic()), client)
            self.assertEqual(server.call_args.args[0].port, 51234)
            self.assertEqual(report["listen"], "127.0.0.1:51234")
            client.json.return_value["engines"][0]["state"] = "error"
            with self.assertRaisesRegex(ValueError, "unexpected engine state"):
                bench.wait_until_ready(process, log, "test-key", args, {}, bench.time.monotonic())

    def test_request_pair_checks_counts_and_greedy_identity(self):
        def sample(request_id):
            response = {
                "id": request_id, "model": "inkling", "engine": "inkling", "object": "text_completion",
                "usage": {"prompt_tokens": 1, "completion_tokens": 3, "total_tokens": 4},
                "choices": [{"index": 0, "text": " Paris", "finish_reason": "length"}],
            }
            rates = {"request_id": request_id, "outcome": "ok", "prompt_tokens": 1,
                     "completion_tokens": 3, "finish_reason": "length"}
            return response, {"requests": [rates]}

        def run(second):
            client = Mock()
            client.json.side_effect = [{"tokens": [12], "length": 1}, *sample("a"), *second]
            report = {"samples": []}
            with patch.object(bench, "disk_reader", return_value=Mock(side_effect=[5, 7, 9, 13])), \
                    patch.object(bench, "swap_pages", return_value={"Swapins": 10, "Swapouts": 0}):
                bench.run_requests(client, 123, {"prompt": "Test", "max_tokens": 3}, 128, report)
            return report

        report = run(sample("b"))
        self.assertEqual([row["attributed_disk_read_bytes"] for row in report["samples"]], [2, 4])
        response, rates = sample("b")
        response["choices"][0]["text"] = " Lyon"
        with self.assertRaisesRegex(ValueError, "outputs differ"):
            run((response, rates))
        response, rates = sample("b")
        response = copy.deepcopy(response)
        response["usage"]["completion_tokens"] = 2
        with self.assertRaisesRegex(ValueError, "unexpected reply usage"):
            run((response, rates))


if __name__ == "__main__":
    unittest.main()
