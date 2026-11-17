#!/usr/bin/env python3
"""Battery arm: attribute-less test functions that never run.

WHAT IT MATCHES: a function is a dead-test finding when ALL hold:
  - it lives in a `*_tests.rs` file or inside a `#[cfg(test)]`-compiled file
    under engine/src or engine/tests;
  - it is not `pub` (test functions never are);
  - it takes no arguments, measured between the signature's opening paren
    and its matching close (so `-> Result<(), E>` stays argless);
  - the lines above it carry no `#[test]`, `#[tokio::test` or `#[rstest`
    attribute — attributes wrapped across lines are joined by walking up
    while brackets stay open, and plain `//` comments between the attribute
    and the fn are skipped;
  - its name has no caller anywhere in engine/src or engine/tests: a caller
    is a word hit on a non-comment line that is not the definition itself.

WHY: rustc's dead_code lint does not flag this shape even under -D warnings
in the ctr-clippy arm (proved: the 95365bbea battery passed with the
attribute-less chunk test present; restored by E14 at da7c59cee). Only a
source-level scan sees the class.

KNOWN LIMITS:
  - fns WITH a test attribute that are excluded by features or #[ignore]
    are not findings (the harness knows them);
  - argless helpers used only by dead tests are not flagged (only the
    test-shaped fn itself);
  - a caller found anywhere in the scanned tree clears the fn, even inside
    another cfg-gated module that may never compile together with it.

Exit codes: 0 clean, 1 findings, 2 self-check or usage failure.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

FN_RE = re.compile(r"^\s*(?:async\s+)?fn\s+([a-z0-9_]+)\s*\(")
TEST_ATTRS = ("#[test", "#[tokio::test", "#[rstest")


def join_continuations(raw: list[str]) -> tuple[list[str], list[int]]:
    """Merge rust `\\`-continued lines; return lines and their first physical no."""
    lines: list[str] = []
    firsts: list[int] = []
    pending: str | None = None
    pending_no = 0
    for no, ln in enumerate(raw, 1):
        if pending is not None:
            ln = pending + ln.lstrip()
        else:
            pending_no = no
        if ln.rstrip().endswith("\\"):
            pending = ln.rstrip()[:-1]
        else:
            lines.append(ln)
            firsts.append(pending_no)
            pending = None
            pending_no = 0
    if pending is not None:
        lines.append(pending)
        firsts.append(pending_no)
    return lines, firsts


def attribute_above(lines: list[str], i: int) -> bool:
    """True when the fn at line i carries a test attribute above it.

    Walks upward while the accumulated block still has an open `[`/`(` (a
    wrapped attribute or argument list read bottom-up), and skips doc
    comments, blanks and plain `//` comments between the attribute and fn.
    Stops at the first line of ordinary code (a closing `}`, a statement),
    which can never belong to this fn's attributes.
    """
    j = i - 1
    balance = 0
    while j >= 0:
        s = lines[j].strip()
        if (
            s == ""
            or s.startswith("///")
            or s.startswith("//!")
            or (s.startswith("//") and not s.startswith("#["))
        ):
            j -= 1
            continue
        balance += s.count("[") + s.count("(") - s.count("]") - s.count(")")
        if any(t in s for t in TEST_ATTRS):
            return True
        # Walking upward, an attribute head still above us shows up as a
        # surplus of closing brackets on the lines below it (balance < 0).
        if s.startswith("#[") or balance < 0:
            j -= 1
            continue
        return False
    return False


def argless(lines: list[str], i: int, name: str) -> bool:
    """True when the fn's argument list, between its parens, is empty.

    Measures only up to the closing paren that matches the signature's
    opening one, so a `()` inside the return type (`Result<(), E>`) is
    never mistaken for the argument list, on one or many lines.
    """
    sig = lines[i]
    k = i
    while "(" not in sig and k + 1 < len(lines) and k < i + 5:
        k += 1
        sig += lines[k]
    m = re.search(r"fn\s+" + re.escape(name) + r"\s*\(", sig)
    if not m:
        return False
    depth = 1  # the opening paren is part of the match itself
    inner: list[str] = []
    for ch in sig[m.end():]:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return "".join(inner).strip() == ""
        elif depth >= 1:
            inner.append(ch)
    return False


def scan_file(path: Path) -> list[tuple[int, str]]:
    raw = path.read_text(errors="replace").splitlines()
    lines, firsts = join_continuations(raw)
    hits: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        m = FN_RE.match(line)
        if not m:
            continue
        name = m.group(1)
        if attribute_above(lines, i):
            continue
        if argless(lines, i, name):
            hits.append((firsts[i], name))
    return hits


def has_caller(name: str, own: Path, roots: list[Path]) -> bool:
    """A caller is a word hit on a non-comment line that is not the definition.

    Comment lines (///, //, and doc-block * continuations) are ignored so a
    prose mention of the name cannot clear a dead test. Any hit in another
    file is a use regardless of shape (the name was referenced by path).
    """
    for root in roots:
        r = subprocess.run(
            ["rg", "-w", "-n", "--no-heading", name,
             "--glob", "!target", str(root)],
            capture_output=True, text=True,
        )
        for hit in r.stdout.splitlines():
            try:
                file_part, _line_no, rest = hit.split(":", 2)
            except ValueError:
                continue
            own_file = Path(own).resolve()
            if Path(file_part).resolve() != own_file:
                return True
            stripped = rest.lstrip()
            if stripped.startswith(("//", "*", "/*")):
                continue
            if re.search(r"fn\s+" + re.escape(name) + r"\s*\(", rest):
                continue
            return True
    return False


def files_to_scan(roots: list[Path]) -> list[Path]:
    out: set[Path] = set()
    for root in roots:
        out.update(root.rglob("*_tests.rs"))
        for p in root.rglob("*.rs"):
            try:
                text = p.read_text(errors="replace")
            except OSError:
                continue
            if "#[cfg(test)]" in text:
                out.add(p)
    return sorted(out)


def scan(roots: list[Path]) -> list[tuple[Path, int, str]]:
    findings = []
    for path in files_to_scan(roots):
        for line_no, name in scan_file(path):
            if name.startswith("_"):
                continue
            if not has_caller(name, path, roots):
                findings.append((path, line_no, name))
    return findings


SELF_CHECK_POSITIVE = """\
#[cfg(test)]
mod pos {
    fn a_dead_test_shaped_fn_never_runs() {
        assert!(true);
    }

    fn a_result_returning_test_still_argless() -> Result<(), String> {
        Ok(())
    }

    fn a_one_line_bodied_test() { assert!(true); }

    #[serial]
    fn serial_alone_does_not_make_a_test() {
        assert!(true);
    }
}
"""

SELF_CHECK_NEGATIVE = """\
#[cfg(test)]
mod neg {
    #[test]
    #[ignore = "needs a GPU (about 4 GiB) and a pack \\
                that spans two lines"]
    fn an_ignored_device_test_still_counts() {
        assert!(true);
    }

    #[tokio::test(flavor = "multi_thread",
                  worker_threads = 2)]
    async fn a_parenthesized_tokio_test_counts() {
        assert!(true);
    }

    // a plain comment between the attribute and the fn
    #[test]
    fn a_comment_does_not_hide_the_attribute() {
        assert!(true);
    }

    pub fn a_pub_helper_is_not_a_test() {}

    fn a_helper_with_a_caller_lives() {
        called_from_elsewhere();
    }
}

fn called_from_elsewhere() {
    a_helper_with_a_caller_lives();
}
"""


SELF_CHECK_POSITIVE_NAMES = [
    "a_dead_test_shaped_fn_never_runs",
    "a_one_line_bodied_test",
    "a_result_returning_test_still_argless",
    "serial_alone_does_not_make_a_test",
]


def self_check() -> int:
    if not shutil.which("rg"):
        print("self-check: rg not found on PATH", file=sys.stderr)
        return 2
    tmp = Path(tempfile.mkdtemp(prefix="dead-test-scan-"))
    try:
        src = tmp / "src"
        (src / "m").mkdir(parents=True)
        (src / "m" / "pos_tests.rs").write_text(SELF_CHECK_POSITIVE)
        (src / "m" / "neg_tests.rs").write_text(SELF_CHECK_NEGATIVE)
        findings = scan([src])
        names = sorted(n for _, _, n in findings)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if names == sorted(SELF_CHECK_POSITIVE_NAMES):
        print(
            "self-check: PASS "
            f"({len(SELF_CHECK_POSITIVE_NAMES)} positives flagged, "
            "negatives clean)"
        )
        return 0
    print(f"self-check: FAIL, findings={names}", file=sys.stderr)
    return 2


def default_roots() -> list[Path]:
    engine = Path(__file__).resolve().parent.parent / "engine"
    roots = [engine / "src", engine / "tests"]
    return [r for r in roots if r.is_dir()]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(__doc__ or "attribute-less test function scanner").splitlines()[0]
    )
    ap.add_argument(
        "--root", type=Path, action="append", dest="roots",
        help="source root to scan (repeatable; default engine/src engine/tests)",
    )
    ap.add_argument(
        "--self-check", action="store_true",
        help="run the embedded positive/negative self-check and exit",
    )
    args = ap.parse_args()
    if args.self_check:
        return self_check()
    roots = args.roots or default_roots()
    if not roots:
        print("no source root found", file=sys.stderr)
        return 2
    rc = self_check()
    if rc != 0:
        return rc
    findings = scan(roots)
    for path, line_no, name in findings:
        try:
            rel = path.relative_to(Path.cwd())
        except ValueError:
            rel = path
        print(f"{rel}:{line_no}  fn {name}")
    if findings:
        print(f"dead-test-scan: {len(findings)} finding(s)", file=sys.stderr)
        return 1
    print("dead-test-scan: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
