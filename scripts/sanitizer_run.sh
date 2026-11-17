#!/usr/bin/env bash
# The compute-sanitizer gate: one listed test per process, per tool. Each test
# runs inside the pinned container image under a chosen sanitizer tool and the
# results land in one table. Not wired into exact-tip-gates.sh: run it per
# landing for .cu/stream/alloc changes, nightly when the card is free, and the
# full table before any default flip.
#
# Usage: sanitizer_run.sh <test-list> [tools...]
#   tools: any of memcheck racecheck initcheck synccheck; default "memcheck synccheck"
# Test-list lines have three '|'-separated fields:  ENV|STATE|NAME
#   ENV    comma-separated KEY=VAL pairs passed to the leg via -e; empty for none
#   STATE  run       - NAME must match exactly one --list entry; that entry runs
#          run-match - every --list entry containing NAME runs
#          skip:R    - listed but skipped, R recorded in the table
#   Blank lines and '#'-comments are ignored.
# Exit status: 0 only when every non-skip row is rc 0 with at least one test
#   passed. A sanitizer finding exits 9 per leg; a failed test exits 101/137;
#   those, NOT_FOUND and AMBIGUOUS rows, and any leg that ran zero tests, all
#   make the script exit 1. skip rows do not affect the exit status.
# Env:
#   SANITIZER_WT        required; worktree root to test
#   SANITIZER_TARGET    required; cargo target dir on the nvme scratch mount
#   SANITIZER_NVME      required; the nvme scratch mount root, bind-mounted into
#                       the containers at the same path
#   SANITIZER_LOCK_DIR  required; lock dir holding gpu.lock and cpu-build*.lock
#   SANITIZER_OUT       optional; receipt dir (default $SANITIZER_NVME/sanitizer-receipts/<utc>)
#   SANITIZER_LEG_SEC   optional; per-leg timeout (default 280)
#   SANITIZER_BUILD_SEC optional; build timeout (default 1800)
# Holds the cpu-build locks (glob order) and then gpu.lock, one open descriptor
# each, all for the whole run: the build leg needs the cpu locks, the card legs
# need gpu.lock.
set -euo pipefail
shopt -s nullglob

usage() { sed -n '2,30p' "$0" >&2; exit 2; }
[ $# -ge 1 ] && [ -f "$1" ] || usage
LIST_FILE=$1; shift
TOOLS=${*:-"memcheck synccheck"}
for t in $TOOLS; do
    case $t in memcheck|racecheck|initcheck|synccheck) ;; *) echo "unknown tool: $t" >&2; usage;; esac
done
WT=${SANITIZER_WT:?SANITIZER_WT is required (worktree root)}
TARGET=${SANITIZER_TARGET:?SANITIZER_TARGET is required (nvme target dir)}
NVME=${SANITIZER_NVME:?SANITIZER_NVME is required (nvme scratch mount root)}
LEG_SEC=${SANITIZER_LEG_SEC:-280}
BUILD_SEC=${SANITIZER_BUILD_SEC:-1800}
LOCK_DIR=${SANITIZER_LOCK_DIR:?SANITIZER_LOCK_DIR is required (gpu.lock and cpu-build*.lock dir)}
OUT=${SANITIZER_OUT:-$NVME/sanitizer-receipts/$(date -u +%Y%m%dT%H%M%SZ)}
mkdir -p "$OUT"
IMAGE=sha256:feff5d69da26e2508144318c62441fa292c0131efe91d6ccccaed4e9d1533d9d
# Keep the digest in step with the one in scripts/exact-tip-gates.sh (A10).
CTR=sanitizer-gate-$$

cleanup() { docker rm -f "$CTR" >/dev/null 2>&1 || true; }
trap cleanup EXIT
# A signal must end the run, not just tidy up and let the next leg start.
on_signal() { trap - EXIT; cleanup; exit 130; }
trap on_signal INT TERM HUP

# Locks: the cpu-build locks in glob order, then the card lock; gpu.lock must
# exist, and an empty set is an error, never a silent lock-free run.
locks=("$LOCK_DIR"/cpu-build*.lock "$LOCK_DIR"/gpu.lock)
[ -e "$LOCK_DIR/gpu.lock" ] || { echo "$LOCK_DIR/gpu.lock is missing" >&2; exit 1; }
[ ${#locks[@]} -gt 0 ] || { echo "no lock files in $LOCK_DIR" >&2; exit 1; }
mkdir -p "$LOCK_DIR"
lock_fds=()
for lock in "${locks[@]}"; do
    exec {lfd}>>"$lock"
    flock -w 3600 "$lfd" || { echo "lock wait timed out: $lock" >&2; exit 1; }
    lock_fds+=("$lfd")   # kept referenced so every descriptor stays open
done
printf '[%s] sanitizer_run start: tools='%s' list=%s out=%s\n' "$(date -u +%FT%TZ)" "$TOOLS" "$LIST_FILE" "$OUT"

# The image's sanitizer version is NOT VERIFIED per the audit trail; record it first.
docker run --pull never --rm -v "$NVME:$NVME" "$IMAGE" compute-sanitizer --version \
    >"$OUT/sanitizer-version.txt" 2>&1 || true
head -1 "$OUT/sanitizer-version.txt"

docker run --pull never --rm --name "$CTR" --cpus 4 --memory 24g -v "$NVME:$NVME" \
    -v "${CARGO_HOME:-$HOME/.cargo}:/host-cargo:ro" \
    -v "${RUSTUP_HOME:-$HOME/.rustup}:/host-rustup:ro" \
    -e CARGO_HOME=/host-cargo -e RUSTUP_HOME=/host-rustup -e CUDA_PATH=/usr/local/cuda-12.6 \
    -e "PATH=/host-cargo/bin:/usr/local/cuda-12.6/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    -e CARGO_TARGET_DIR="$TARGET" -w "$WT/engine" "$IMAGE" \
    timeout -s KILL "$BUILD_SEC" \
    cargo test --locked --offline --release --features native-cuda --lib --no-run \
    >"$OUT/build.log" 2>&1 || { tail -5 "$OUT/build.log"; exit 1; }
EXE=$(grep -oE "${TARGET}[^ )]*deps/inference_engine-[0-9a-f]+" "$OUT/build.log" | tail -1)
[ -n "$EXE" ] || { echo "no test executable path in $OUT/build.log" >&2; exit 1; }

docker run --pull never --rm --gpus all --cpus 2 --memory 8g -v "$NVME:$NVME" \
    -e INFERENCE_CUDA_REQUIRE_DEVICE=1 "$IMAGE" timeout -s KILL 120 "$EXE" --list \
    >"$OUT/tests.txt" 2>/dev/null
echo "listed tests: $(wc -l <"$OUT/tests.txt")"

TABLE=$OUT/table.tsv
FAILS=0
printf 'tool\tenv\tstate\ttest\trc\tverdict\tsummary\n' >"$TABLE"
row() { printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$@" | tee -a "$TABLE"; }

for tool in $TOOLS; do
    while IFS='|' read -r env state name; do
        case "$env$name$state" in \#*|'') continue;; esac
        [ -n "$name" ] || continue
        if [[ $state == skip:* ]]; then
            row "$tool" "$env" "$state" "$name" - SKIP "listed, not run"
            continue
        fi
        if [[ $state == run ]]; then
            # Anchor the name so a substring of another test cannot match.
            esc=${name//./\\.}
            mapfile -t hits < <(grep -E "(^|::)${esc}: test$" "$OUT/tests.txt" | sed 's/: test$//')
        else
            mapfile -t hits < <(grep -F -- "$name" "$OUT/tests.txt" | sed 's/: test$//')
        fi
        if [ ${#hits[@]} -eq 0 ]; then
            row "$tool" "$env" "$state" "$name" - NOT_FOUND "no --list entry matches"
            FAILS=$((FAILS+1)); continue
        fi
        if [[ $state == run ]] && [ ${#hits[@]} -gt 1 ]; then
            row "$tool" "$env" "$state" "$name" - AMBIGUOUS "${#hits[@]} list entries match"
            FAILS=$((FAILS+1)); continue
        fi
        for test in "${hits[@]}"; do
            env_flags=()
            [ -n "$env" ] && IFS=',' read -ra pairs <<<"$env" \
                && for p in "${pairs[@]}"; do env_flags+=(-e "$p"); done
            log=$OUT/$tool-${test//::/-}.log
            rc=0
            docker run --pull never --rm --gpus all --cpus 8 --memory 16g \
                --memory-swap 16g --name "$CTR" -v "$NVME:$NVME" -e INFERENCE_CUDA_REQUIRE_DEVICE=1 \
                "${env_flags[@]}" "$IMAGE" timeout -s KILL "$LEG_SEC" \
                compute-sanitizer --tool "$tool" --error-exitcode 9 "$EXE" \
                --exact "$test" --include-ignored --test-threads=1 >"$log" 2>&1 || rc=$?
            summary=$(grep -E 'ERROR SUMMARY|RACECHECK SUMMARY|INITCHECK SUMMARY|SYNCCHECK SUMMARY|[0-9]+ passed; [0-9]+ failed' "$log" | tail -1)
            verdict=PASS
            if [ "$rc" -ne 0 ]; then
                verdict=FAIL; FAILS=$((FAILS+1))
            elif grep -qE 'running 0 tests|(^|; )0 passed' "$log"; then
                # A listed #[ignore] test without --include-ignored would land here as a
                # clean-looking non-run; zero tests is never a pass.
                verdict=FAIL; FAILS=$((FAILS+1))
                summary="zero tests ran; ${summary:-no summary line}"
            fi
            row "$tool" "$env" run "$test" "$rc" "$verdict" "${summary:-no summary line}"
        done
    done <"$LIST_FILE"
done
echo "table: $TABLE"
[ "$FAILS" -eq 0 ] || exit 1
