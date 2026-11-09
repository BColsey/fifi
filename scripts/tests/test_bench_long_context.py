"""CPU-only tests for long-context evidence and its HTTP contract."""

import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import bench_long_context as bench  # noqa: E402


def event(value):
    return ("data: " + json.dumps(value) + "\n\n").encode()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def reply(self, payload, content_type="application/json"):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path == "/v1/system":
            self.server.system_calls += 1
            limit = self.server.context_limit
            if self.server.change_context and self.server.system_calls > 1:
                limit -= 1
            data = {"engines": [{"model": "test-model", "state": "ready",
                                 "context_limit": limit}]}
        elif self.path == "/v1/recent-requests":
            data = {"requests": [{"request_id": "test-request", "model": "test-model",
                    "status": 200, "outcome": "ok", "stream": True,
                    "finish_reason": self.server.finish_reason,
                    "prompt_tokens": self.server.prompt_count,
                    "completion_tokens": 14, "decode_tokens_per_second": 17.5}]}
        else:
            self.send_error(404)
            return
        self.reply(json.dumps(data).encode())

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/v1/token/encode":
            if "text" in body:
                length = len(body["text"])
            else:
                length = len(body["messages"][0]["content"]) + 17
            self.reply(json.dumps({"length": length, "tokens": [0] * length}).encode())
        elif self.path == "/v1/chat/completions":
            self.server.chat_count += 1
            self.server.auth = self.headers.get("Authorization")
            text = body["messages"][0]["content"]
            values = re.findall(r"ARCHIVE MARKER (?:MAPLE|COBALT|FJORD): (\d{6})\.", text)
            if len(values) != 3:
                self.send_error(400)
                return
            answer = "|".join(values)
            if self.server.wrong_answer:
                answer = "|".join(values[:2] + ["wrong"])
            self.server.prompt_count = len(text) + 17
            usage = {"prompt_tokens": self.server.prompt_count + self.server.usage_delta,
                     "completion_tokens": 14}
            if self.server.reused is not None:
                usage["reused_tokens"] = self.server.reused
            payload = b"".join([
                event({"id": "test-request", "choices": [{"delta": {"content": answer[:5]}}]}),
                event({"id": "test-request", "choices": [{"delta": {"content": answer[5:]}}]}),
                event({"id": "test-request", "choices": [{"delta": {},
                       "finish_reason": self.server.finish_reason}], "usage": usage}),
            ])
            if not self.server.incomplete:
                payload += b"data: [DONE]\n\n"
            self.reply(payload, "text/event-stream")
        else:
            self.send_error(404)


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name) / "result.json"
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.context_limit = 10000
        self.server.finish_reason = "stop"
        self.server.wrong_answer = False
        self.server.usage_delta = 0
        self.server.reused = 0
        self.server.chat_count = 0
        self.server.system_calls = 0
        self.server.change_context = False
        self.server.incomplete = False
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 5)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def cli(self, *extra, env=None):
        return subprocess.run(
            [sys.executable, str(ROOT / "scripts/bench_long_context.py"),
             "--url", f"http://127.0.0.1:{self.server.server_port}", "--model", "test-model",
             "--prompt-tokens", "3000", "--output", str(self.output), *extra],
            capture_output=True, text=True, timeout=10, env=env,
        )

    def test_real_http_flow_counts_prompt_and_verifies_all_markers(self):
        env = {**os.environ, "TEST_FIFI_KEY": "synthetic-key"}
        result = self.cli("--api-key-env", "TEST_FIFI_KEY", env=env)
        self.assertEqual(result.returncode, 0, result.stderr)
        evidence = json.loads(self.output.read_text())
        self.assertEqual(evidence["state"], "complete")
        self.assertTrue(evidence["recall_passed"])
        self.assertEqual(evidence["completion_tokens"], 14)  # two chunks are not two tokens
        self.assertEqual(evidence["reused_tokens"], 0)
        self.assertEqual(evidence["server_decode_tokens_per_second"], 17.5)
        self.assertTrue(3000 <= evidence["prompt_tokens"] <= 3128)
        prompt = bench.prompt_text(evidence["filler_repeats"], evidence["nonce"],
                                   tuple(evidence["expected_answer"].split("|")))
        self.assertEqual(hashlib.sha256(prompt.encode()).hexdigest(), evidence["prompt_sha256"])
        self.assertEqual(self.server.auth, "Bearer synthetic-key")
        self.assertNotIn("synthetic-key", self.output.read_text())
        self.assertFalse(self.output.with_name("result.json.partial").exists())

    def test_wrong_last_marker_is_recorded_as_failure(self):
        self.server.wrong_answer = True
        result = self.cli()
        self.assertEqual(result.returncode, 2, result.stderr)
        evidence = json.loads(self.output.read_text())
        self.assertFalse(evidence["recall_passed"])
        self.assertEqual(evidence["state"], "complete")

    def test_large_prompt_budgets_leave_space_for_output(self):
        for target, reply in ((131072, 8192), (257900, 4096)):
            with self.subTest(target=target):
                self.server.context_limit = 262144
                result = self.cli("--prompt-tokens", str(target), "--max-tokens", str(reply))
                self.assertEqual(result.returncode, 0, result.stderr)
                evidence = json.loads(self.output.read_text())
                self.assertTrue(target <= evidence["prompt_tokens"] <= target + 128)
                self.assertLessEqual(evidence["prompt_tokens"] + reply, self.server.context_limit)
                self.output.unlink()

    def test_truncated_answer_is_not_a_pass_even_if_its_text_matches(self):
        self.server.finish_reason = "length"
        result = self.cli()
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertFalse(json.loads(self.output.read_text())["recall_passed"])

    def test_inconsistent_usage_and_unconfirmed_cold_reuse_are_rejected(self):
        for delta, reused in ((1, 0), (0, 7), (0, None)):
            with self.subTest(delta=delta, reused=reused):
                self.server.usage_delta, self.server.reused = delta, reused
                result = self.cli()
                self.assertEqual(result.returncode, 1, result.stderr)
                partial = self.output.with_name("result.json.partial")
                self.assertEqual(json.loads(partial.read_text())["state"], "failed")
                self.assertFalse(self.output.exists())
                partial.unlink()

    def test_reserved_output_must_fit_before_starting_a_chat(self):
        self.server.context_limit = 3050
        result = self.cli("--max-tokens", "64")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(self.server.chat_count, 0)
        self.assertIn("exceed the context limit", result.stderr)

    def test_engine_context_change_blocks_the_chat_request(self):
        self.server.change_context = True
        result = self.cli()
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(self.server.chat_count, 0)
        self.assertIn("context changed", result.stderr)

    def test_incomplete_stream_is_not_recorded_as_a_completed_result(self):
        self.server.incomplete = True
        result = self.cli()
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertFalse(self.output.exists())
        partial = self.output.with_name("result.json.partial")
        self.assertEqual(json.loads(partial.read_text())["state"], "failed")

    def test_existing_output_is_never_overwritten(self):
        self.output.write_text("existing")
        result = self.cli()
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(self.output.read_text(), "existing")
        self.assertEqual(self.server.chat_count, 0)

    def test_missing_credential_and_nonfinite_timeout_are_rejected(self):
        with mock.patch.dict(os.environ, {"TEST_FIFI_MISSING": ""}):
            result = self.cli("--api-key-env", "TEST_FIFI_MISSING")
        self.assertEqual(result.returncode, 1, result.stderr)
        result = self.cli("--timeout", "nan")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertFalse(self.output.exists())

    def test_preflight_requests_have_shorter_timeouts_than_generation(self):
        client = bench.Client("http://127.0.0.1:18792", None, 21600)
        with mock.patch.object(bench.urllib.request, "urlopen",
                               side_effect=lambda *_args, **_kw: io.BytesIO(b"{}")) as opened:
            client.json("/v1/system")
            client.json("/v1/token/encode", {"text": "a"})
            client.request("/v1/chat/completions", {})
        self.assertEqual([call.kwargs["timeout"] for call in opened.call_args_list], [5, 180, 21600])


if __name__ == "__main__":
    unittest.main()
