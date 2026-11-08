import http.server
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest

SCRIPT = Path(__file__).parents[1] / 'bench_p2p_decode.py'


class DecodeDriverTest(unittest.TestCase):
    def test_three_requests_use_matching_server_telemetry(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / 'server.log'
            output = Path(directory) / 'result.json'
            log.write_text('')

            class Handler(http.server.BaseHTTPRequestHandler):
                count = 0

                def log_message(self, *_):
                    pass

                def send_json(self, value):
                    body = json.dumps(value).encode()
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Content-Length', str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

                def do_GET(self):
                    self.send_json({'engines': [{'model': 'fifi-q36', 'state': 'ready',
                                                 'context_limit': 131072}]})

                def do_POST(self):
                    length = int(self.headers['Content-Length'])
                    request = json.loads(self.rfile.read(length))
                    self.assertions(request)
                    type(self).count += 1
                    rid = f'test-{type(self).count}'
                    row = {'request_id': rid, 'outcome': 'ok', 'reused_tokens': 0,
                           'prompt_tokens': 25, 'completion_tokens': 128,
                           'time_to_first_token_ms': 600, 'decode_tokens_per_second': 24.75}
                    with log.open('a') as f:
                        f.write(json.dumps(row) + '\n')
                    self.send_json({'id': rid, 'usage': {'prompt_tokens': 25, 'completion_tokens': 128},
                                    'choices': [{'message': {'content': '1\n2\n3\n4'},
                                                 'finish_reason': 'length'}]})

                @staticmethod
                def assertions(request):
                    assert request['temperature'] == 0 and request['max_tokens'] == 128

            server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                subprocess.run([sys.executable, str(SCRIPT), '--samples', '3', '--server-log', str(log),
                                '--output', str(output), '--url',
                                f'http://127.0.0.1:{server.server_port}'], check=True,
                               stdout=subprocess.DEVNULL)
            finally:
                server.shutdown()
                server.server_close()
                thread.join()
            result = json.loads(output.read_text())
            self.assertEqual(len(result['samples']), 3)
            self.assertEqual(result['median_decode_tokens_per_second'], 24.75)
            self.assertEqual([s['request_id'] for s in result['samples']],
                             ['test-1', 'test-2', 'test-3'])


if __name__ == '__main__':
    unittest.main()
