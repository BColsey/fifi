"""Check shared ignore rules without inheriting this checkout's local excludes."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]


class GitignoreTests(unittest.TestCase):
    def setUp(self) -> None:
        scratch_value = os.environ.get("TMPDIR")
        if not scratch_value:
            self.fail("TMPDIR must name an existing directory")
        scratch = Path(scratch_value).resolve()
        if not scratch.is_dir():
            self.fail("TMPDIR must name an existing directory")
        temporary = tempfile.TemporaryDirectory(prefix="gitignore-test-", dir=scratch)
        self.addCleanup(temporary.cleanup)
        self.repo = Path(temporary.name)
        self.environment = dict(os.environ)
        self.environment["GIT_CONFIG_NOSYSTEM"] = "1"
        self.environment["GIT_CONFIG_GLOBAL"] = os.devnull
        shutil.copy2(ROOT / ".gitignore", self.repo / ".gitignore")
        result = self.git("init", "-q")
        self.assertEqual(result.returncode, 0, result.stderr)

    def git(self, *arguments: str, stdin: str | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-c", f"core.excludesFile={os.devnull}", *arguments],
            cwd=self.repo,
            env=self.environment,
            input=stdin,
            capture_output=True,
            text=True,
            check=False,
        )

    def ignored(self, paths: list[str]) -> set[str]:
        result = self.git("check-ignore", "--no-index", "--stdin", stdin="\n".join(paths) + "\n")
        self.assertIn(result.returncode, (0, 1), result.stderr)
        return set(result.stdout.splitlines())

    def test_instructions_can_be_added_without_force(self) -> None:
        for name in ("AGENTS.md", "CLAUDE.md"):
            shutil.copy2(ROOT / name, self.repo / name)
        result = self.git("add", "--", "AGENTS.md", "CLAUDE.md")
        self.assertEqual(result.returncode, 0, result.stderr)
        result = self.git("ls-files")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(set(result.stdout.splitlines()), {"AGENTS.md", "CLAUDE.md"})

    def test_generated_artifacts_are_ignored(self) -> None:
        paths = [
            "target/debug/enginesrv", "engine/target/debug/enginesrv",
            "engine/vendor/example/target/debug/lib.rlib", "target-host/debug/test",
            "engine/target-native/debug/test", ".worktrees/lane/.git", "tmp/nvcc.o",
            "models/model.safetensors", "weights/model.gguf", "downloads/model.part",
            "profiles/run.nsys-rep", "profiles/run.qdrep", "profiles/run.ncu-rep",
            ".cache/compiler/data", "engine/.pytest_cache/state", ".ruff_cache/state",
            ".mypy_cache/state", ".venv/bin/python", ".env", "run.log",
            "engine/__pycache__/module.pyc", "web/node_modules/vite/package.json",
            "web/dist/index.html", "web/.test-dist/api.test.js",
            ".claude/settings.local.json", ".claude/scheduled_tasks.lock",
            ".claude/scheduled_tasks.json", ".claude/routines/.state/state.json",
            ".claude/worktrees/lane/.git", ".claude/checkpoints/state.json",
            ".claude/mailbox/message.json", ".claude/agent-registry.json",
            ".claude/agent-memory-local/memory.md", ".claude/first-run",
            ".claude/assistant-daemon-state.json", "AGENTS.local.md",
        ]
        # The frontend owns its existing output rules separately.
        (self.repo / "web").mkdir()
        shutil.copy2(ROOT / "web/.gitignore", self.repo / "web/.gitignore")
        self.assertEqual(self.ignored(paths), set(paths))

    def test_source_fixtures_and_shared_settings_stay_visible(self) -> None:
        paths = [
            "AGENTS.md", "CLAUDE.md", ".claude/settings.json",
            ".claude/commands/review.md", ".cargo/config.toml", "engine/Cargo.lock",
            "engine/src/kernels/cuda/exl3.cu", "engine/tests/fixtures/new.safetensors",
            "engine/tests/fixtures/small.gguf", "assets/config.json",
            "scripts/tests/test_gitignore.py", "qualification/token-forcing/driver.py",
            "qualification/mtp-acceptance/collector.py",
        ]
        self.assertEqual(self.ignored(paths), set())


class ScratchEnvironmentTests(unittest.TestCase):
    def test_missing_tmpdir_is_rejected(self) -> None:
        for value in (None, ""):
            with self.subTest(value=value), mock.patch.dict(os.environ):
                if value is None:
                    os.environ.pop("TMPDIR", None)
                else:
                    os.environ["TMPDIR"] = value
                case = GitignoreTests("test_generated_artifacts_are_ignored")
                try:
                    with self.assertRaisesRegex(AssertionError, "TMPDIR must name"):
                        case.setUp()
                finally:
                    case.doCleanups()


if __name__ == "__main__":
    unittest.main()
