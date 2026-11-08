"""Exercise local model ordering and routing with loopback fake backends."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SCRIPT = Path(__file__).resolve().parents[1] / "web-model-router.mjs"


class Backend(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def reply(self):
        if self.path == "/v1/models" and self.server.unavailable:
            self.send_error(503)
            return
        if self.path == "/v1/models":
            value = {"object": "list", "data": [{"id": self.server.model}]}
        elif self.path.startswith("/v1/sampler-trace?"):
            value = {
                "backend": self.server.model,
                "auth_ok": self.headers.get("authorization")
                == "Bearer " + self.server.key,
                "forwarded_model_header": self.headers.get("x-fifi-trace-model"),
            }
        else:
            value = {
                "backend": self.server.model,
                "auth_ok": self.headers.get("authorization")
                == "Bearer " + self.server.key,
            }
        body = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.reply()

    def do_POST(self):
        self.rfile.read(int(self.headers.get("content-length", "0")))
        self.reply()


class ModelRouterTests(unittest.TestCase):
    def setUp(self):
        node = shutil.which("node")
        if not node:
            self.fail("Node.js is required to test the model router")
        temp = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        primary = root / "primary"
        secondary = root / "secondary"
        primary.mkdir()
        secondary.mkdir()
        (primary / "api-key").write_text("primary-key\n")
        (secondary / "api-key").write_text("secondary-key\n")
        self.backends = []
        for model, key in (
            ("fifi-q36", "primary-key"),
            ("fifi-q36-2080ti", "secondary-key"),
        ):
            server = ThreadingHTTPServer(("127.0.0.1", 0), Backend)
            server.model = model
            server.key = key
            server.unavailable = False
            threading.Thread(target=server.serve_forever, daemon=True).start()
            self.addCleanup(server.server_close)
            self.addCleanup(server.shutdown)
            self.backends.append(server)
        source = SCRIPT.read_text()
        for original, server in (
            ("http://127.0.0.1:18791", self.backends[0]),
            ("http://127.0.0.1:18793", self.backends[1]),
        ):
            self.assertEqual(source.count(original), 1)
            source = source.replace(original, f"http://127.0.0.1:{server.server_port}")
        copy = root / "router.mjs"
        copy.write_text(source)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        env = dict(
            os.environ,
            FIFI_PRIMARY_RUN=str(primary),
            FIFI_SECONDARY_RUN=str(secondary),
            FIFI_MODEL_ROUTER_PORT=str(port),
        )
        self.proc = subprocess.Popen(
            [node, str(copy)],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.addCleanup(self.stop_router)
        self.base = f"http://127.0.0.1:{port}"
        for _ in range(60):
            if self.proc.poll() is not None:
                self.fail("model router exited before accepting requests")
            try:
                self.get("/v1/models")
                break
            except OSError:
                time.sleep(0.05)
        else:
            self.fail("model router did not start")

    def stop_router(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait(timeout=5)

    def get(self, path, headers=None):
        request = urllib.request.Request(self.base + path, headers=headers or {})
        with urllib.request.urlopen(request, timeout=4) as response:
            return json.load(response)

    def post(self, model):
        body = json.dumps({"model": model}).encode()
        request = urllib.request.Request(
            self.base + "/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=4) as response:
            return json.load(response)

    def test_primary_is_first_and_model_specific_routes_still_work(self):
        data = self.get("/v1/models")["data"]
        self.assertEqual(
            [model["id"] for model in data], ["fifi-q36", "fifi-q36-2080ti"]
        )
        self.assertEqual(
            self.get("/v1/system"), {"backend": "fifi-q36", "auth_ok": True}
        )
        self.assertEqual(
            self.post("fifi-q36"), {"backend": "fifi-q36", "auth_ok": True}
        )
        self.assertEqual(
            self.post("fifi-q36-2080ti"),
            {"backend": "fifi-q36-2080ti", "auth_ok": True},
        )
        self.assertEqual(self.post("unknown"), {"backend": "fifi-q36", "auth_ok": True})

    def test_primary_placeholder_stays_first_if_its_backend_is_unavailable(self):
        self.backends[0].unavailable = True
        self.assertEqual(
            [model["id"] for model in self.get("/v1/models")["data"]],
            ["fifi-q36", "fifi-q36-2080ti"],
        )

    def test_trace_lookup_uses_the_reply_model_even_when_ids_collide(self):
        path = "/v1/sampler-trace?id=shared-id"
        for model in ("fifi-q36", "fifi-q36-2080ti"):
            with self.subTest(model=model):
                result = self.get(path, {"X-Fifi-Trace-Model": model})
                self.assertEqual(result["backend"], model)
                self.assertTrue(result["auth_ok"])
                self.assertIsNone(result["forwarded_model_header"])
        self.assertEqual(self.get(path)["backend"], "fifi-q36")
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.get(path, {"X-Fifi-Trace-Model": "unknown-model"})
        self.assertEqual(error.exception.code, 400)

    def test_model_list_fails_when_both_backends_are_unavailable(self):
        for backend in self.backends:
            backend.unavailable = True
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.get("/v1/models")
        self.assertEqual(error.exception.code, 502)


if __name__ == "__main__":
    unittest.main()
