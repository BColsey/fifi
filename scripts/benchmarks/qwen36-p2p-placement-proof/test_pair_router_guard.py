import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('reserve_router_test', Path(__file__).with_name('reserve_pair_attention.py'))
r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)


class RouterPollGuardTest(unittest.TestCase):
    def run_case(self, mutation=None):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            node = root / 'node'; node.write_text('node')
            script = root / 'router.mjs'; script.write_text('router')
            engine = root / 'enginesrv'; engine.write_text('engine')
            start = root / 'start.sh'; start.write_text(f'exec "{engine}" --config ignored\n')
            rows = [(18791, 35156, '111'), (35156, 18791, '222')]
            data = {'rows': rows, 'router_fds': {'222'}, 'server_fds': {'111'},
                    'router_identity': (10, node, [str(node).encode(), str(script).encode()]),
                    'server_identity': (20, engine, [str(engine).encode()]), 'router_pid': 55, 'server_pid': 66}
            if mutation: mutation(data)
            calls = {'tcp': 0, 'router_identity': 0}
            def tcp():
                calls['tcp'] += 1
                return data.get('rows_after', data['rows']) if calls['tcp'] > 1 else data['rows']
            def identity(pid):
                if pid == 55:
                    calls['router_identity'] += 1
                    value = data.get('router_identity_after', data['router_identity']) if calls['router_identity'] > 1 else data['router_identity']
                else:
                    value = data['server_identity']
                if isinstance(value, Exception): raise value
                return value
            def fds(pid):
                value = data['router_fds'] if pid == 55 else data['server_fds']
                if isinstance(value, Exception): raise value
                return value
            def state(unit):
                return {'ActiveState': 'active', 'MainPID': str(data['router_pid'] if unit == r.ROUTER_UNIT else data['server_pid'])}
            with patch.object(r, 'ROUTER_EXE', node), patch.object(r, 'ROUTER_SCRIPT', script), patch.object(r, 'START', start), \
                 patch.object(r, 'tcp_rows', side_effect=tcp), patch.object(r, 'state', side_effect=state), \
                 patch.object(r, 'process_identity', side_effect=identity), patch.object(r, 'fd_sockets', side_effect=fds), \
                 patch.object(r, 'clients', return_value='ESTAB 0 0 127.0.0.1:18791 127.0.0.1:35156'):
                return r.confirmed_router_clients(66)

    def test_exact_both_socket_ends_accepted(self):
        self.assertEqual(self.run_case()[0]['router_inode'], '222')

    def test_unknown_client_or_unpaired_fails(self):
        for mutate in (lambda d: d.update(router_fds={'999'}),
                       lambda d: d.update(rows=[(18791, 35156, '111')]),
                       lambda d: d.update(server_fds={'999'}),
                       lambda d: d.update(router_fds=PermissionError('unreadable'))):
            with self.subTest(mutate=mutate), self.assertRaises((RuntimeError, PermissionError)):
                self.run_case(mutate)

    def test_changed_socket_set_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, 'changed during verification'):
            self.run_case(lambda data: data.update(rows_after=[]))

    def test_changed_process_identity_is_fatal_even_if_socket_set_also_changes(self):
        def mutate(data):
            start, exe, cmdline = data['router_identity']
            data.update(rows_after=[], router_identity_after=(start + 1, exe, cmdline))
        with self.assertRaisesRegex(RuntimeError, 'process identity changed') as failure:
            self.run_case(mutate)
        self.assertNotIsInstance(failure.exception, r.TransientSnapshotRace)

    def test_wrong_router_or_server_identity_fails(self):
        for mutate in (lambda d: d.update(router_identity=(10, Path('/wrong'), [b'/wrong', b'/wrong'])),
                       lambda d: d.update(server_identity=(20, Path('/wrong'), [b'/wrong'])),
                       lambda d: d.update(router_pid=99),
                       lambda d: d.update(server_pid=99)):
            with self.subTest(mutate=mutate), self.assertRaises((RuntimeError, FileNotFoundError)):
                self.run_case(mutate)

    def test_endpoint_snapshot_mismatch_fails(self):
        with self.assertRaisesRegex(RuntimeError, 'ss address'):
            r.ss_peer_ports('ESTAB 0 0 127.0.0.1:18791 10.1.1.1:35156')
        self.assertEqual(r.ss_peer_ports('ESTAB 0 0 127.0.0.1:18791 127.0.0.1:35156'), [35156])

    def test_three_observations_require_ready_and_zero_utilization(self):
        samples=[]
        with patch.object(r, 'clients', return_value='poll'), patch.object(r, 'confirmed_router_clients', return_value=[{'client_port': 3}]) as checked, \
             patch.object(r, 'ready', return_value=True), patch.object(r, 'gpu_values', return_value={gpu:0 for gpu in r.PAIR}), \
             patch.object(r.time, 'sleep'):
            self.assertTrue(r.pair_idle(observations=samples, allow_confirmed_router=True, server_pid=66))
        self.assertEqual(checked.call_count, 3)
        self.assertEqual(len(samples), 3)

    def test_ss_mismatch_retries_full_snapshot_then_accepts_three_stable_observations(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); node = root/'node'; node.write_text('node')
            script = root/'router.mjs'; script.write_text('router')
            engine = root/'enginesrv'; engine.write_text('engine')
            start = root/'start.sh'; start.write_text(f'exec "{engine}" --config ignored\n')
            rows = [(18791, 35156, '111'), (35156, 18791, '222')]
            good = 'ESTAB 0 0 127.0.0.1:18791 127.0.0.1:35156'
            different = 'ESTAB 0 0 127.0.0.1:18791 127.0.0.1:35157'
            calls = [good, different] + [good, good] * 3
            def identity(pid):
                return (10, node, [str(node).encode(), str(script).encode()]) if pid == 55 else (20, engine, [str(engine).encode()])
            def state(unit):
                return {'ActiveState':'active', 'MainPID':'55' if unit == r.ROUTER_UNIT else '66'}
            observations = []
            with patch.object(r, 'ROUTER_EXE', node), patch.object(r, 'ROUTER_SCRIPT', script), patch.object(r, 'START', start), \
                 patch.object(r, 'tcp_rows', return_value=rows) as tcp, patch.object(r, 'state', side_effect=state), \
                 patch.object(r, 'process_identity', side_effect=identity) as process, \
                 patch.object(r, 'fd_sockets', side_effect=lambda pid: {'222'} if pid == 55 else {'111'}) as fds, \
                 patch.object(r, 'clients', side_effect=calls), patch.object(r, 'ready', return_value=True), \
                 patch.object(r, 'gpu_values', return_value={gpu:0 for gpu in r.PAIR}), patch.object(r.time, 'sleep'):
                self.assertTrue(r.pair_idle(observations=observations, allow_confirmed_router=True, server_pid=66))
            self.assertEqual(observations[0]['event'], 'socket_snapshot_retry')
            self.assertEqual(len(observations), 4)
            self.assertEqual(tcp.call_count, 8)
            self.assertEqual(process.call_count, 16)
            self.assertEqual(fds.call_count, 8)

    def test_a_race_resets_prior_quiet_observations(self):
        stable = [{'client_port': 3}]
        with patch.object(r, 'clients', return_value='poll'), \
             patch.object(r, 'confirmed_router_clients', side_effect=[stable, r.TransientSnapshotRace('changed'), stable, stable, stable]) as verify, \
             patch.object(r, 'ready', return_value=True), patch.object(r, 'gpu_values', return_value={gpu:0 for gpu in r.PAIR}), \
             patch.object(r.time, 'sleep'):
            self.assertTrue(r.pair_idle(allow_confirmed_router=True, server_pid=66))
            self.assertEqual(verify.call_count, 5)

    def test_repeated_snapshot_races_exhaust_existing_idle_bound(self):
        with patch.object(r, 'clients', return_value='poll'), \
             patch.object(r, 'confirmed_router_clients', side_effect=r.TransientSnapshotRace('TCP snapshot and ss endpoints disagree')) as verify, \
             patch.object(r, 'gpu_values', return_value={gpu:0 for gpu in r.PAIR}), patch.object(r.time, 'sleep'):
            self.assertFalse(r.pair_idle(allow_confirmed_router=True, server_pid=66))
            self.assertEqual(verify.call_count, 18)

    def test_foreign_and_unreadable_identities_fail_without_retry(self):
        for error in (RuntimeError('client socket is not owned by expected router'), PermissionError('unreadable')):
            with self.subTest(error=error), patch.object(r, 'clients', return_value='poll'), \
                 patch.object(r, 'confirmed_router_clients', side_effect=error) as verify, \
                 patch.object(r, 'gpu_values', return_value={gpu:0 for gpu in r.PAIR}), patch.object(r.time, 'sleep'):
                with self.assertRaises(type(error)):
                    r.pair_idle(allow_confirmed_router=True, server_pid=66)
                self.assertEqual(verify.call_count, 1)

    def test_router_exception_still_refuses_busy_or_not_ready(self):
        for util, ready in (('1', True), ('0', False)):
            with self.subTest(util=util, ready=ready), patch.object(r, 'clients', return_value='poll'), \
                 patch.object(r, 'confirmed_router_clients', return_value=[{'client_port': 3}]), \
                 patch.object(r, 'ready', return_value=ready), \
                 patch.object(r, 'gpu_values', return_value={gpu:int(util) for gpu in r.PAIR}), patch.object(r.time, 'sleep'):
                self.assertFalse(r.pair_idle(allow_confirmed_router=True, server_pid=66))

    def test_default_refuses_any_connected_client(self):
        with patch.object(r, 'clients', return_value='connected'), patch.object(r, 'gpu_values', return_value={gpu:0 for gpu in r.PAIR}), \
             patch.object(r, 'ready', return_value=True), patch.object(r.time, 'sleep'):
            self.assertFalse(r.pair_idle())

    def test_final_socket_check_retries_only_transient_races(self):
        stable = [{'client_port': 3}]
        observations = []
        with patch.object(r, 'checked_clients', side_effect=[r.TransientSnapshotRace('changed'), stable]) as check, \
             patch.object(r.time, 'sleep'):
            self.assertEqual(r.final_checked_clients(True, 66, observations), stable)
            self.assertEqual(check.call_count, 2)
            self.assertEqual(observations[0]['event'], 'final_socket_snapshot_retry')
        with patch.object(r, 'checked_clients', side_effect=r.TransientSnapshotRace('changing')) as check, \
             patch.object(r.time, 'sleep'):
            self.assertIsNone(r.final_checked_clients(True, 66))
            self.assertEqual(check.call_count, 3)
        with patch.object(r, 'checked_clients', side_effect=PermissionError('foreign socket')) as check, \
             patch.object(r.time, 'sleep'):
            with self.assertRaises(PermissionError):
                r.final_checked_clients(True, 66)
            self.assertEqual(check.call_count, 1)

if __name__ == '__main__': unittest.main()
