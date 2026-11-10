import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import attention_device as a
import reserve_pair_attention as r


def result(target=512):
    return {'state': 'complete', 'target_prompt_tokens': target, 'prompt_tokens': target + 3,
            'completion_tokens': 8, 'reused_tokens': 0, 'request_id': 'req1',
            'expected_answer': '123456|234567|345678', 'answer': '123456|234567|345678',
            'finish_reason': 'stop', 'recall_passed': True, 'first_visible_seconds': 3.0,
            'server_prefill_ms': 2000.0}


def server_log(target=512):
    return json.dumps({'endpoint': '/v1/chat/completions', 'request_id': 'req1',
                       'status': 200, 'outcome': 'ok', 'prompt_tokens': target + 3,
                       'completion_tokens': 8, 'reused_tokens': 0, 'prefill_ms': 2000.0}) + '\n'


class LongRunnerTests(unittest.TestCase):
    def test_result_requires_matching_cold_server_telemetry(self):
        good = result()
        measured = a.validate_result(good, server_log(), 512)
        self.assertTrue(measured['recall_passed'])
        self.assertEqual([item['correct'] for item in measured['marker_checks']], [True, True, True])
        self.assertEqual(measured['server_first_batch_tokens_per_second'], 257.5)
        for key, bad in [('reused_tokens', 1), ('prompt_tokens', True), ('request_id', 'missing'),
                         ('first_visible_seconds', 0), ('server_prefill_ms', 1000),
                         ('recall_passed', False)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                a.validate_result({**good, key: bad}, server_log(), 512)
        with self.assertRaises(ValueError):
            a.validate_result(good, server_log().replace('"reused_tokens": 0', '"reused_tokens": 1'), 512)

    def test_incorrect_recall_is_a_recorded_failure_not_a_pass(self):
        bad = {**result(), 'answer': 'wrong', 'recall_passed': False}
        measured = a.validate_result(bad, server_log(), 512)
        self.assertFalse(measured['recall_passed'])
        self.assertEqual([item['correct'] for item in measured['marker_checks']], [False, False, False])
        bad['answer'] = '123456|000000|345678'
        self.assertEqual([item['correct'] for item in a.validate_result(bad, server_log(), 512)['marker_checks']],
                         [True, False, True])
        with self.assertRaises(ValueError):
            a.validate_result({**bad, 'recall_passed': True}, server_log(), 512)

    def test_smoke_receipt_binds_result_inputs_guard_and_restoration(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            campaign = root / 'pair-reserved-abc'
            out = campaign / 'attention'
            out.mkdir(parents=True)
            info = {'binary_sha256': 'pinned'}
            (campaign / 'complete.json').write_text(json.dumps({'restored': True, 'result_type': 'p2p_correctness_tests'}))
            (campaign / 'restore.json').write_text(json.dumps({'restored': True}))
            (out / 'candidate-engine.log').write_text(server_log())
            (out / 'recall-512.json').write_text(json.dumps(result()))
            (out / 'summary.json').write_text(json.dumps({'mode': 'smoke', 'inputs': info, 'candidate_warmup': True,
                'server_log_sha256': a.sha(out / 'candidate-engine.log'),
                'recall_sha256': {'512': a.sha(out / 'recall-512.json')}}))
            sample = {'builders': [], 'other_jobs': [], 'unexpected': [], 'states_ok': True,
                      'clients': False, 'listeners': False, 'non_target_utilization': {'other': 0},
                      'pair_free_mib': {gpu: 1100 for gpu in a.PAIR_VISIBLE.split(',')}}
            (campaign / 'guard.jsonl').write_text(json.dumps(sample) + '\n')
            with patch.object(a, 'ROOT', root), patch.object(a.bench, 'check_path'):
                self.assertEqual(len(a.validate_smoke_receipt(str(campaign), info)), 64)
                with self.assertRaises(ValueError):
                    a.validate_smoke_receipt(str(campaign), {'binary_sha256': 'other'})
                (campaign / 'restore.json').write_text(json.dumps({'restored': False}))
                with self.assertRaises(ValueError):
                    a.validate_smoke_receipt(str(campaign), info)

    def test_driver_accepts_only_matching_completed_exit(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            log = root / 'engine.log'
            log.write_text(server_log())
            def invoke(cmd, **kwargs):
                Path(cmd[cmd.index('--output') + 1]).write_text(json.dumps(result()))
                return MagicMock(returncode=0)
            with patch.object(a.subprocess, 'run', side_effect=invoke) as call:
                measured = a.run_recall(512, root, {'safe': 'environment'}, log)
            self.assertTrue(measured['recall_passed'])
            cmd = call.call_args.args[0]
            self.assertIn('--api-key-env', cmd)
            self.assertEqual(cmd[cmd.index('--prompt-tokens') + 1], '512')
            self.assertEqual(a.TARGETS['full'], (8192, 32768, 131072))

    def test_progress_relay_excludes_unmarked_and_incomplete_lines(self):
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / 'runner.log'
            log.write_text('private detail\nPROGRESS started\nPROGRESS part')
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                offset = r.relay_progress(log, 0)
                self.assertEqual(output.getvalue(), 'PROGRESS started\n')
                with log.open('a') as stream:
                    stream.write('ial\n')
                r.relay_progress(log, offset)
            self.assertEqual(output.getvalue(), 'PROGRESS started\nPROGRESS partial\n')

    def test_discovers_all_current_assistant_owners(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for pid, command, start in ((10, 'claude', 123), (20, 'claude', 456), (30, 'other', 789)):
                owner = root / str(pid)
                owner.mkdir()
                (owner / 'comm').write_text(command + '\n')
                fields = ['S'] + ['0'] * 18 + [str(start)]
                (owner / 'stat').write_text(f'{pid} ({command}) ' + ' '.join(fields))
            self.assertEqual(r.current_assistant_owners(root), {10: 123, 20: 456})


if __name__ == '__main__':
    unittest.main()
