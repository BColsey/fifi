from pathlib import Path
import unittest
from unittest.mock import patch
import scope_guard as s

class ScopeGuardTest(unittest.TestCase):
    def data(self, scope):
        return {'Id': scope.unit, 'LoadState': 'loaded', 'ActiveState': 'active',
                'InvocationID': 'a' * 32, 'Delegate': 'no',
                'ControlGroup': f'/user.slice/user-{s.os.getuid()}.slice/user@{s.os.getuid()}.service/app.slice/{scope.unit}'}

    def test_uninitialized_loaded_scope_retries_without_weakening_identity(self):
        scope = s.Scope(); data = self.data(scope)
        initial = {**data, 'ActiveState': 'inactive', 'ControlGroup': '', 'InvocationID': ''}
        with patch.object(scope, 'query', side_effect=[initial, data]):
            self.assertFalse(scope.observe())
            self.assertIsNone(scope.invocation)
            self.assertTrue(scope.observe())
        for bad in ({**initial, 'ActiveState': 'active'},
                    {**initial, 'ActiveState': 'activating'},
                    {**initial, 'ControlGroup': data['ControlGroup']},
                    {**initial, 'InvocationID': data['InvocationID']},
                    {**initial, 'Delegate': 'yes'}):
            fresh = s.Scope(); bad['Id'] = fresh.unit
            with self.subTest(bad=bad), patch.object(fresh, 'query', return_value=bad):
                with self.assertRaises(RuntimeError): fresh.observe()
        # Once pinned, disappearance of identity must never be treated as startup.
        with patch.object(scope, 'query', return_value=initial):
            with self.assertRaises(RuntimeError): scope.observe()

    def test_cleanup_does_not_release_for_pending_scope_after_launcher_exits(self):
        scope = s.Scope(); data = self.data(scope)
        pending = {**data, 'ActiveState': 'inactive', 'ControlGroup': '', 'InvocationID': ''}
        child = unittest.mock.MagicMock(); child.poll.return_value = 1
        with patch.object(scope, 'query', side_effect=[pending, data]) as query, \
             patch.object(scope, 'populated', return_value=False), patch.object(s.time, 'sleep'):
            scope.cleanup(child)
        self.assertEqual(query.call_count, 2)
        self.assertEqual(scope.invocation, data['InvocationID'])
        child.wait.assert_called()

    def test_identity_pins_scope_and_refuses_changed_invocation_and_delegation(self):
        scope = s.Scope(); data = self.data(scope)
        with patch.object(scope, 'query', return_value=data):
            self.assertTrue(scope.observe())
        for change in ({'InvocationID': 'b' * 32}, {'Delegate': 'yes'}, {'ControlGroup': '/wrong'}):
            with self.subTest(change=change), patch.object(scope, 'query', return_value={**data, **change}):
                with self.assertRaises(RuntimeError): scope.observe()

    def test_kill_refuses_replaced_scope(self):
        scope = s.Scope(); data = self.data(scope)
        with patch.object(scope, 'query', return_value=data): scope.observe()
        with patch.object(scope, 'query', return_value={**data, 'InvocationID': 'b' * 32}), \
             patch.object(s.subprocess, 'run') as command:
            with self.assertRaises(RuntimeError): scope.kill('SIGTERM')
            command.assert_not_called()

    def test_contains_refuses_outside_scope(self):
        scope = s.Scope(); scope.cgroup = '/user.slice/example.scope'
        with patch.object(Path, 'read_text', return_value='0::/user.slice/other.scope\n'):
            self.assertFalse(scope.contains(123))
        with patch.object(Path, 'read_text', return_value='0::/user.slice/example.scope/child\n'):
            self.assertTrue(scope.contains(123))

if __name__ == '__main__': unittest.main()
