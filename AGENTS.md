# fifi inference engine — project instructions

`main` is the integration branch (Dan, 2026-11-07): it carries reviewed,
test-green work, and new behaviour stays default-off until it is qualified.
Merged, measured, qualified and enabled-by-default are separate states.
Performance numbers characterize work; they do not block integration.

## Checkout and integration

- Before work, read `~/AGENTS.md` and `AGENTS.local.md` in the canonical
  checkout, when present. Use `git worktree list --porcelain` to locate
  the primary checkout. Local instructions do not accompany new worktrees.
- Do not develop in the canonical checkout. Use a dedicated scratch worktree
  and the host's documented storage layout:
  `git -C <canonical-checkout> worktree add <scratch>/<base-commit>/<task>/wt -b task/<name> <base>`.
- Do not switch, reset, rebase, or dirty the canonical checkout; it stays on `main`.
- Advancing `main` is fast-forward-only. A coordinator session may
  fast-forward a slice without per-merge approval when the slice is
  reviewed, rebased onto current `main`, and green on its exact tip
  (Dan, 2026-11-07). Before the fast-forward, verify the hash, subject,
  base and diffstat, and check that `main` has not moved. Then run
  `git merge --ff-only <tip>` in the canonical checkout.
- A default flip still needs its qualification evidence or an explicit
  decision by Dan.
- Outward synchronization to other hosts or remotes is on HOLD.
- Preserve other lanes' modified and untracked files. Check local instructions
  before cleanup or integration. While the canonical checkout has a modified
  `README.md`, slices must not change it. Put such text in a `doc/` page instead.

## Execution and coordination

- Before work in a new worktree, check that it has the applicable project
  instructions. This `AGENTS.md` is tracked and accompanies new worktrees
  based on commits that include it.
- Keep build output, caches, downloads and profiles in the task's scratch
  directory. Set `TMPDIR`, `TMP`, `TEMP` and `CARGO_TARGET_DIR` there.
  Ignore rules prevent accidental tracking; they do not reclaim disk space.
- Before testing, check the target, platform and features. Enumerate the
  expected tests. Treat zero matching tests as a failed check.
- Use the established device runner. Run CUDA device tests with
  `--test-threads=1`. Preserve container paths when binaries depend on
  `CARGO_MANIFEST_DIR` or other paths embedded during compilation.
- Before reserving GPU time, finish required builds. Check input files,
  scratch permissions, disk space and declared VRAM against admission limits.
- Test the measurement driver on the CPU where possible. If the driver
  requires hardware, use a short admitted device run before the full campaign.
- After each run, check required artifacts and cleanup. A zero exit code
  alone does not prove that the run produced valid results.
- Use the existing coordinator queue. Record the owner, next action and
  wake condition for each blocked task. Prefer completion events to polling.
  Stop recurring monitors when quota exhaustion prevents further work.
- State what changed before repeating a failed run. If an infrastructure
  failure recurs, repair the shared runner before another attempt.
- Each handoff must include the exact commit, completed checks, evidence
  paths, next command and any blocker. Report merged, measured, qualified
  and enabled-by-default status separately.
