"""Opt-in real user-scope test using a sealed no-network fixture and temp lock."""

import fcntl
import hashlib
import json
import os
from pathlib import Path
import select
import shutil
import subprocess
import tempfile
import unittest

from scripts.p2p_concurrent_guard.host_attestation import _seal_pinned, sealed_gate


WRAPPER = r"""
import json,os,sys
from pathlib import Path
gate_source_fd,program_fd,lease_fd,signal_fd,lock_path,scratch,expected,marker = sys.argv[1:]
gate_source_fd,program_fd,lease_fd,signal_fd = map(int,(gate_source_fd,program_fd,lease_fd,signal_fd))
src = os.pread(gate_source_fd, 1 << 20, 0)
scope = {'__name__':'gated_fixture'}
exec(compile(src, 'sealed-gate', 'exec'), scope)
scope['SCRATCH'] = Path(scratch)
old_lock, old_program = scope['verify_lock'], scope['verify_program']
scope['verify_lock'] = lambda fd: old_lock(fd, Path(lock_path))
scope['verify_program'] = lambda fd: old_program(fd, expected)
groups=[line.split(':',2)[2] for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::')]
print(json.dumps({'pid':os.getpid(), 'groups':groups}),flush=True)
options = scope['CompanionOptions'](scratch, str(Path(scratch)/'receipt.jsonl'), 1, 10)
scope['execute'](signal_fd, lease_fd, program_fd, options, timeout=10)
"""


@unittest.skipUnless(
    os.environ.get("FIFI_SCOPE_REAL_TEST") == "1", "explicit temporary scope test"
)
@unittest.skipUnless(hasattr(os, "memfd_create"), "Linux memfd required")
class GateScopeTest(unittest.TestCase):
    def test_sealed_gated_fixture_waits_for_parent_and_execs_in_scope(self):
        if not shutil.which("systemd-run") or not shutil.which("systemctl"):
            self.skipTest("systemd tools unavailable")
        try:
            subprocess.run(
                ["systemctl", "--user", "show-environment"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
            self.skipTest("user manager unavailable")
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            marker = directory / "fixture-ran"
            lock_path = directory / "lease"
            lock_path.touch(mode=0o600)
            source = directory / "no-network-fixture.py"
            source.write_text(
                f"from pathlib import Path\nPath({str(marker)!r}).write_text('ok')\n"
            )
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            program = _seal_pinned(source, digest)
            gate = sealed_gate(
                Path(__file__).resolve().parents[1]
                / "p2p_concurrent_guard/gated_companion.py"
            )
            lock_fd = os.open(lock_path, os.O_RDWR | os.O_CLOEXEC)
            read_fd, write_fd = os.pipe2(os.O_CLOEXEC)
            child = None
            unit = f"fifi-gate-test-{os.getpid()}-{os.urandom(8).hex()}.scope"
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                args = [
                    "systemd-run",
                    "--user",
                    "--scope",
                    "--quiet",
                    "--unit=" + unit,
                    "/usr/bin/python3",
                    "-I",
                    "-c",
                    WRAPPER,
                    str(gate),
                    str(program),
                    str(lock_fd),
                    str(read_fd),
                    str(lock_path),
                    root,
                    digest,
                    str(marker),
                ]
                child = subprocess.Popen(
                    args,
                    pass_fds=(gate, program, lock_fd, read_fd),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                os.close(read_fd)
                read_fd = None
                self.assertTrue(
                    select.select([child.stdout], [], [], 10)[0],
                    "scope child did not report",
                )
                report = json.loads(child.stdout.readline())
                self.assertEqual(len(report["groups"]), 1)
                self.assertEqual(Path(report["groups"][0]).name, unit)
                self.assertFalse(marker.exists(), "fixture ran before parent signalled")
                self.assertIsNone(child.poll())
                os.write(write_fd, b"\x01")
                os.close(write_fd)
                write_fd = None
                _, stderr = child.communicate(timeout=10)
                self.assertEqual(child.returncode, 0, stderr)
                self.assertEqual(marker.read_text(), "ok")
                group = Path("/sys/fs/cgroup") / report["groups"][0].lstrip("/")
                if group.exists():
                    self.assertIn("populated 0", (group / "cgroup.events").read_text())
            finally:
                if child is not None and child.poll() is None:
                    subprocess.run(
                        ["systemctl", "--user", "stop", unit],
                        timeout=5,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                    )
                    child.kill()
                    child.communicate(timeout=5)
                for fd in (read_fd, write_fd, lock_fd, gate, program):
                    if fd is not None:
                        os.close(fd)


if __name__ == "__main__":
    unittest.main()
