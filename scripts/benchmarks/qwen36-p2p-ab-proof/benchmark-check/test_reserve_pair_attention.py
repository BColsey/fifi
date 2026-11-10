import contextlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import reserve_pair_attention as r


class PairTest(unittest.TestCase):
    def test_clients_blocks_all_non_listen_tcp_states(self):
        listen = 'LISTEN 0 128 127.0.0.1:18791 0.0.0.0:*'
        for status in ('ESTAB', 'SYN-RECV', 'CLOSE-WAIT', 'TIME-WAIT', 'FIN-WAIT-1'):
            row = f'{status} 0 0 127.0.0.1:18791 127.0.0.1:50000'
            with self.subTest(status=status), patch.object(r.subprocess, 'check_output', return_value=listen+'\n'+row):
                self.assertEqual(r.clients(), row)
        with patch.object(r.subprocess, 'check_output', return_value=listen):
            self.assertEqual(r.clients(), '')
        with patch.object(r.subprocess, 'check_output', return_value='bad row'):
            with self.assertRaises(RuntimeError): r.clients()

    def test_pair_idle_requires_three_ready_zero_util_client_free_observations(self):
        zero = {gpu:0 for gpu in r.PAIR}
        with patch.object(r, 'clients', side_effect=['client', '', '', '']), \
             patch.object(r, 'gpu_values', return_value=zero), patch.object(r, 'ready', return_value=True), \
             patch.object(r.time, 'sleep'):
            observations=[]
            self.assertTrue(r.pair_idle(observations))
            self.assertEqual(len(observations), 4)
        for client, util, readiness in (('client', zero, True), ('', {**zero, r.TARGET:1}, True), ('', zero, False)):
            with self.subTest(client=client, util=util, ready=readiness), \
                 patch.object(r, 'clients', return_value=client), patch.object(r, 'gpu_values', return_value=util), \
                 patch.object(r, 'ready', return_value=readiness), patch.object(r.time, 'sleep'):
                self.assertFalse(r.pair_idle())

    def test_unit_fingerprint_detects_fragment_and_manager_changes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); frag = root/'unit'; frag.write_text('unit')
            launcher = root/'start'; launcher.write_text('start')
            drop = root/'dropin'; drop.write_text('drop')
            raw = f'FragmentPath={frag}\nDropInPaths={drop}\nNeedDaemonReload=no\nExecStart={{ path={launcher} ; argv[]={launcher} ; ignore_errors=no ; start_time=[now] ; pid=1 }}\nEnvironment=SECRET=secret\n'
            def output(args, **kwargs):
                return b'MANAGER=one' if 'show-environment' in args else raw
            with patch.object(r, 'START', launcher), patch.object(r.subprocess, 'check_output', side_effect=output):
                first = r.unit_fingerprint(); self.assertNotIn('secret', json.dumps(first))
                drop.write_text('changed'); self.assertNotEqual(first, r.unit_fingerprint())
            with patch.object(r, 'START', launcher), patch.object(r.subprocess, 'check_output', side_effect=lambda args, **kw: b'MANAGER=two' if 'show-environment' in args else raw):
                self.assertNotEqual(first, r.unit_fingerprint())
            for bad in (raw.replace('NeedDaemonReload=no', 'NeedDaemonReload=yes'), raw+'EnvironmentFiles=/tmp/other\n',
                        raw+'ExecStartPre=unexpected\n', raw.replace('ignore_errors=no', 'ignore_errors=yes'),
                        raw.replace(f'argv[]={launcher}', 'argv[]=/bin/other')):
                with self.subTest(bad=bad), patch.object(r, 'START', launcher), patch.object(r.subprocess, 'check_output', return_value=bad), self.assertRaises(RuntimeError):
                    r.unit_fingerprint()

    def test_restoration_waits_for_stop_settlement_then_confirms_pair_ready(self):
        exe = Path('/fake/original'); fp={'unit':'same'}
        states = [{'ActiveState':'deactivating','Job':'1'}, {'ActiveState':'deactivating','Job':'1'},
                  {'ActiveState':'inactive','Job':''}, {'ActiveState':'active','MainPID':'57','Job':''}]
        with patch.object(r, 'state', side_effect=states), patch.object(r, 'unchanged', return_value=True), \
             patch.object(r, 'unit_fingerprint', return_value=fp), patch.object(r.guard, 'processes', return_value=[]), \
             patch.object(r, 'port_open', return_value=False), patch.object(r, 'listeners', return_value=''), \
             patch.object(r, 'clients', return_value=''), patch.object(r, 'pair_owner', return_value=True), \
             patch.object(r.os.path, 'samefile', return_value=True), patch.object(r, 'ready', return_value=True), \
             patch.object(r.subprocess, 'run') as command, patch.object(r.time, 'sleep'):
            self.assertEqual(r.restore({exe:'good'}, fp, 55), (True, 'original pair service ready'))
            self.assertIn('start', command.call_args.args[0])

    def test_restoration_refuses_changed_unit_or_busy_pair_socket(self):
        cases = ('unit', 'gpu', 'listener', 'client')
        for case in cases:
            with self.subTest(case=case):
                now = [0]
                def sleep(n): now[0] += n
                with patch.object(r, 'state', return_value={'ActiveState':'inactive','Job':''}), \
                     patch.object(r, 'unchanged', return_value=True), \
                     patch.object(r, 'unit_fingerprint', return_value={'unit':'changed' if case=='unit' else 'same'}), \
                     patch.object(r.guard, 'processes', return_value=[(r.TARGET,123)] if case=='gpu' else []), \
                     patch.object(r, 'port_open', return_value=False), \
                     patch.object(r, 'listeners', return_value='listener' if case=='listener' else ''), \
                     patch.object(r, 'clients', return_value='client' if case=='client' else ''), \
                     patch.object(r.time, 'monotonic', side_effect=lambda:now[0]), patch.object(r.time, 'sleep', side_effect=sleep), \
                     patch.object(r.subprocess, 'run') as command:
                    ok, _ = r.restore({Path('/fake'):'good'}, {'unit':'same'}, 55)
                    self.assertFalse(ok); command.assert_not_called()

    def test_restoration_waits_past_60_second_time_wait(self):
        now=[0]; started=[]; exe=Path('/fake/original'); fp={'unit':'same'}
        def sleep(n): now[0] += n
        def state(_unit):
            return {'ActiveState':'active' if started else 'inactive', 'MainPID':'57', 'Job':''}
        def command(args, **kwargs):
            started.append(now[0]); return MagicMock(returncode=0)
        def clients(): return 'TIME-WAIT 0 0 127.0.0.1:18791 127.0.0.1:50000' if now[0] < 65 else ''
        with patch.object(r, 'state', side_effect=state), patch.object(r, 'unchanged', return_value=True), \
             patch.object(r, 'unit_fingerprint', return_value=fp), patch.object(r.guard, 'processes', return_value=[]), \
             patch.object(r, 'port_open', return_value=False), patch.object(r, 'listeners', return_value=''), \
             patch.object(r, 'clients', side_effect=clients), patch.object(r, 'pair_owner', return_value=True), \
             patch.object(r.os.path, 'samefile', return_value=True), patch.object(r, 'ready', return_value=True), \
             patch.object(r.subprocess, 'run', side_effect=command), patch.object(r.time, 'monotonic', side_effect=lambda:now[0]), \
             patch.object(r.time, 'sleep', side_effect=sleep):
            self.assertEqual(r.restore({exe:'good'}, fp, 55), (True, 'original pair service ready'))
            self.assertEqual(started, [65])

    def test_stop_timeout_and_interrupt_settle_before_restart(self):
        for error in (subprocess.TimeoutExpired('stop',45), SystemExit(143)):
            with self.subTest(error=error), tempfile.TemporaryDirectory() as td:
                root=Path(td); exe=root/'exe'; exe.write_text('engine')
                start=root/'start'; start.write_text('start'); config=root/'config'; config.write_text('config')
                build=root/'env'; build.write_text('env'); key=root/'secret'; key.write_text('secret')
                status={'stop':False,'checks':0,'started':False}
                def state(unit):
                    if unit == r.SECONDARY: return {'ActiveState':'active','MainPID':'66','Job':''}
                    if not status['stop']: return {'ActiveState':'active','MainPID':'55','Job':''}
                    status['checks'] += 1
                    if status['checks'] <= 2: return {'ActiveState':'deactivating','MainPID':'55','Job':'1'}
                    return {'ActiveState':'active' if status['started'] else 'inactive', 'MainPID':'57', 'Job':''}
                def command(args, **kwargs):
                    if 'stop' in args: status['stop']=True; raise error
                    if 'start' in args: status['started']=True
                    return MagicMock(returncode=0)
                def values(query, selected): return {gpu:(8000 if query=='memory.used' else 0) for gpu in selected}
                patches=[patch.object(r,'ROOT',root),patch.object(r,'START',start),patch.object(r,'CONFIG',config),
                         patch.object(r,'BUILD_ENV',build),patch.object(r,'API_KEY',key),
                         patch.object(r,'lock_check'),patch.object(r,'preflight',return_value={}),
                         patch.object(r.guard,'assistant_children',return_value=set()),patch.object(r.guard,'assistant_activity',return_value=[]),
                         patch.object(r.guard,'builders',return_value=[]),patch.object(r.guard,'processes',return_value=[]),
                         patch.object(r,'state',side_effect=state),patch.object(r,'pair_owner',return_value=True),
                         patch.object(r,'other_owner',return_value=True),patch.object(r,'pair_idle',return_value=True),
                         patch.object(r,'gpu_values',side_effect=values),patch.object(r,'ready',return_value=True),
                         patch.object(r,'unit_fingerprint',return_value={'u':'x'}),patch.object(r.os,'readlink',return_value=str(exe)),
                         patch.object(r.os.path,'samefile',return_value=True),patch.object(r,'clients',return_value=''),
                         patch.object(r,'listeners',return_value=''),patch.object(r,'port_open',return_value=False),
                         patch.object(r.subprocess,'run',side_effect=command),patch.object(r.time,'sleep'),
                         patch.object(r.signal,'signal'),patch.object(r.scope_guard,'Scope')]
                with contextlib.ExitStack() as stack:
                    for item in patches: stack.enter_context(item)
                    with self.assertRaises(type(error)): r.execute()
                self.assertTrue(status['started']); self.assertGreaterEqual(status['checks'], 4)
                receipt=next(root.glob('pair-reserved-*/restore.json'))
                self.assertTrue(json.loads(receipt.read_text())['restored'])
                self.assertNotIn('secret', (receipt.parent/'inputs.json').read_text())

    def test_post_stop_waits_past_60_second_time_wait_before_launch(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); exe=root/'exe'; exe.write_text('engine')
            start=root/'start'; start.write_text('start'); config=root/'config'; config.write_text('config')
            build=root/'env'; build.write_text('env'); key=root/'secret'; key.write_text('secret')
            now=[0]; stopped=[]
            def sleep(n): now[0] += n
            def state(unit): return {'ActiveState':'active','MainPID':'55' if unit==r.PRIMARY else '66','Job':''}
            def values(query, selected):
                value = 8000 if query=='memory.used' else 12000 if query=='memory.free' else 0
                return {gpu:value for gpu in selected}
            def clients():
                return 'TIME-WAIT 0 0 127.0.0.1:18791 127.0.0.1:50000' if stopped and now[0] < 65 else ''
            def command(args, **kwargs):
                if 'stop' in args: stopped.append(True)
                return MagicMock(returncode=0)
            patches=[patch.object(r,'ROOT',root),patch.object(r,'START',start),patch.object(r,'CONFIG',config),
                     patch.object(r,'BUILD_ENV',build),patch.object(r,'API_KEY',key),patch.object(r,'lock_check'),
                     patch.object(r,'preflight',side_effect=[{}, RuntimeError('verified drain')]),
                     patch.object(r.guard,'assistant_children',return_value=set()),patch.object(r.guard,'assistant_activity',return_value=[]),
                     patch.object(r.guard,'builders',return_value=[]),patch.object(r.guard,'processes',return_value=[]),
                     patch.object(r,'state',side_effect=state),patch.object(r,'pair_owner',return_value=True),
                     patch.object(r,'other_owner',return_value=True),patch.object(r,'pair_idle',return_value=True),
                     patch.object(r,'gpu_values',side_effect=values),patch.object(r,'ready',return_value=True),
                     patch.object(r,'unit_fingerprint',return_value={'u':'x'}),patch.object(r.os,'readlink',return_value=str(exe)),
                     patch.object(r.os.path,'samefile',return_value=True),patch.object(r,'clients',side_effect=clients),
                     patch.object(r,'listeners',return_value=''),patch.object(r,'port_open',return_value=False),
                     patch.object(r.subprocess,'run',side_effect=command),patch.object(r.subprocess,'Popen') ,
                     patch.object(r,'restore',return_value=(True,'ready')),patch.object(r.signal,'signal'),
                     patch.object(r.time,'monotonic',side_effect=lambda:now[0]), patch.object(r.time,'sleep',side_effect=sleep),
                     patch.object(r.scope_guard,'Scope')]
            with contextlib.ExitStack() as stack:
                mocks=[stack.enter_context(item) for item in patches]
                with self.assertRaisesRegex(RuntimeError, 'verified drain'): r.execute()
            self.assertGreaterEqual(now[0],65)
            mocks[24].assert_not_called()

    def test_cancellation_drains_scope_before_restoration(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); exe=root/'exe'; exe.write_text('engine')
            start=root/'start'; start.write_text('start'); config=root/'config'; config.write_text('config')
            build=root/'env'; build.write_text('env'); key=root/'secret'; key.write_text('secret')
            events=[]; child=MagicMock(); child.poll.return_value=None
            scope=MagicMock(); scope.unit='unit'; scope.cgroup='/fake'; scope.invocation='a'*32
            scope.command.side_effect=lambda cmd:cmd
            scope.cleanup.side_effect=lambda value: events.append('scope-drained')
            def restore(*args): events.append('restore'); return True, 'ready'
            def state(unit): return {'ActiveState':'active','MainPID':'55' if unit==r.PRIMARY else '66','Job':''}
            def values(query, selected):
                value = 8000 if query=='memory.used' else 12000 if query=='memory.free' else 0
                return {gpu:value for gpu in selected}
            patches=[patch.object(r,'ROOT',root),patch.object(r,'START',start),patch.object(r,'CONFIG',config),
                     patch.object(r,'BUILD_ENV',build),patch.object(r,'API_KEY',key),patch.object(r,'lock_check'),
                     patch.object(r,'preflight',return_value={}),patch.object(r.guard,'assistant_children',return_value=set()),
                     patch.object(r.guard,'assistant_activity',side_effect=[[],[],[],SystemExit(143)]),
                     patch.object(r.guard,'builders',return_value=[]),patch.object(r.guard,'processes',return_value=[]),
                     patch.object(r,'state',side_effect=state),patch.object(r,'pair_owner',return_value=True),
                     patch.object(r,'other_owner',return_value=True),patch.object(r,'pair_idle',return_value=True),
                     patch.object(r,'gpu_values',side_effect=values),patch.object(r,'ready',return_value=True),
                     patch.object(r,'unit_fingerprint',return_value={'u':'x'}),patch.object(r.os,'readlink',return_value=str(exe)),
                     patch.object(r.os.path,'samefile',return_value=True),patch.object(r,'clients',return_value=''),
                     patch.object(r,'listeners',return_value=''),patch.object(r,'port_open',return_value=False),
                     patch.object(r.subprocess,'run'),patch.object(r.subprocess,'Popen',return_value=child),
                     patch.object(r,'restore',side_effect=restore),patch.object(r.signal,'signal'),
                     patch.object(r.scope_guard,'Scope',return_value=scope),
                     patch.dict(r.os.environ, {'FIFI_P2P_BUILD_RECEIPT':'/receipt','FIFI_CONTROL_BUILD_RECEIPT':'/control-receipt','FIFI_PEER_PROOF':'/peer-proof'})]
            with contextlib.ExitStack() as stack:
                for item in patches: stack.enter_context(item)
                with self.assertRaises(SystemExit): r.execute()
            self.assertEqual(events, ['scope-drained','restore'])
            receipt=next(root.glob('pair-reserved-*/restore.json'))
            self.assertTrue(json.loads(receipt.read_text())['restored'])
            self.assertFalse((receipt.parent/'complete.json').exists())
            self.assertFalse((receipt.parent/'tmp').exists())

    def test_profile_pre_and_post_guard_bracket_before_completion(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); exe=root/'exe'; exe.write_text('engine')
            start=root/'start'; start.write_text('start'); config=root/'config'; config.write_text('config')
            build=root/'env'; build.write_text('env'); key=root/'secret'; key.write_text('secret')
            stopped=[]; events=[]; child=MagicMock(); child.poll.return_value=0; child.wait.return_value=0
            scope=MagicMock(); scope.unit='unit'; scope.cgroup='/fake'; scope.invocation='a'*32
            scope.command.side_effect=lambda cmd:cmd
            scope.cleanup.side_effect=lambda value:events.append('scope-drained')
            def state(unit):
                if unit == r.SECONDARY: return {'ActiveState':'active','MainPID':'66','Job':''}
                return {'ActiveState':'inactive' if stopped else 'active','MainPID':'0' if stopped else '55','Job':''}
            def values(query, selected):
                value=8000 if query=='memory.used' else 12000 if query=='memory.free' else 0
                return {gpu:value for gpu in selected}
            def command(args, **kwargs):
                if 'stop' in args: stopped.append(True)
                return MagicMock(returncode=0)
            def popen(args, **kwargs):
                out=Path(args[-1]); out.mkdir()
                (out/'summary.json').write_text(json.dumps({'mode':'profile','profile_evidence':{
                    'event_start_utc_ns':2_000_000_000,'event_end_utc_ns':2_100_000_000}}))
                return child
            def restore(*args): events.append('restore'); return True,'ready'
            info={'nsys':'/fake/nsys','nsys_sha256':'x','importer_sha256':'y'}
            patches=[patch.object(r,'ROOT',root),patch.object(r,'START',start),patch.object(r,'CONFIG',config),
                     patch.object(r,'BUILD_ENV',build),patch.object(r,'API_KEY',key),patch.object(r,'lock_check'),
                     patch.object(r,'preflight',return_value=info),patch.object(r.guard,'assistant_children',return_value=set()),
                     patch.object(r.guard,'assistant_activity',return_value=[]),patch.object(r.guard,'builders',return_value=[]),
                     patch.object(r.guard,'processes',return_value=[]),patch.object(r,'state',side_effect=state),
                     patch.object(r,'pair_owner',return_value=True),patch.object(r,'other_owner',return_value=True),
                     patch.object(r,'pair_idle',return_value=True),patch.object(r,'gpu_values',side_effect=values),
                     patch.object(r,'ready',return_value=True),patch.object(r,'unit_fingerprint',return_value={'u':'x'}),
                     patch.object(r.os,'readlink',return_value=str(exe)),patch.object(r.os.path,'samefile',return_value=True),
                     patch.object(r,'clients',return_value=''),patch.object(r,'listeners',return_value=''),
                     patch.object(r,'port_open',return_value=False),patch.object(r.subprocess,'run',side_effect=command),
                     patch.object(r.subprocess,'Popen',side_effect=popen),patch.object(r,'restore',side_effect=restore),
                     patch.object(r.signal,'signal'),patch.object(r.time,'time',side_effect=[1.0,3.0]),
                     patch.object(r.scope_guard,'Scope',return_value=scope),
                     patch.dict(r.os.environ, {'FIFI_PAIR_SMOKE_RECEIPT':'/smoke','FIFI_P2P_BUILD_RECEIPT':'/receipt','FIFI_CONTROL_BUILD_RECEIPT':'/control-receipt','FIFI_PEER_PROOF':'/peer-proof'})]
            with contextlib.ExitStack() as stack:
                for item in patches: stack.enter_context(item)
                r.execute('profile')
            receipt=next(root.glob('pair-reserved-*/complete.json'))
            self.assertTrue(json.loads(receipt.read_text())['restored'])
            samples=[json.loads(line) for line in (receipt.parent/'guard.jsonl').read_text().splitlines()]
            self.assertEqual([x['time'] for x in samples],[1.0,3.0])
            self.assertEqual(events,['scope-drained','restore'])

    def test_full_requires_same_input_smoke(self):
        with patch.object(r.scope_guard,'check_manager'), patch.object(r.attention,'preflight',return_value={}), \
             patch.object(r.os,'access',return_value=True), patch.dict(r.os.environ,{},clear=True):
            with self.assertRaisesRegex(ValueError,'requires completed'):
                r.preflight('full')

if __name__ == '__main__': unittest.main()
