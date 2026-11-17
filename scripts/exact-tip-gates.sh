#!/usr/bin/env bash
# Exact-tip gates for one commit, from clean trees and fresh target dirs.
#
# The host arms alone miss breakage that only the container (native-cuda) or the
# Mac (native-mlx) build sees, so this battery runs every arm a handoff needs:
# fmt, host clippy, host lib tests, published_text, the qualification inventory,
# container clippy, container test build and a device-free lib test run (release,
# native-cuda, no GPU), and a Mac
# native-mlx clippy. The commit is materialized as a detached worktree — a clean
# checkout of the exact hash, never a working copy — and every Linux arm reads
# that worktree; the Mac arm reads a `git archive` stream of the same commit
# (kept so no worktree metadata reaches the Mac). Both local target dirs are
# deleted at exit; only the small per-arm logs and the receipt file are kept.
#
# Usage:
#   GATES_SCRATCH=<dir> scripts/exact-tip-gates.sh <commit> [--mac=always|auto|never] [--list]
#
# Environment:
#   GATES_SCRATCH      required; the run directory is created and cleaned inside it
#   GATES_LOCK         required; path of the shared CPU-build lock file
#   GATES_MAC_HOST     required when the Mac arm runs; the Mac build host (ssh alias)
#   GATES_MAC_BASE     required when the Mac arm runs; absolute scratch base on the
#                      Mac (not /tmp; validated before any rm -rf under it)
#   GATES_MAC_MAX_SEC  optional; LOCAL time limit around the Mac ssh (default
#                      1800). It bounds the wait, not the remote build: on a
#                      timeout kill the remote shell's own EXIT trap still
#                      removes its directory when the build ends.
#   GATES_ARM_MAX_SEC  optional; per-arm time limit for every Linux arm (default 3600)
#   GATES_LOCK_WAIT    optional; seconds to wait for the shared lock (default 7200;
#                      a wait that times out refuses the run with rc 5)
#   GATES_MIN_FREE_GB  optional; free-disk floor on the scratch filesystem (default 15)
#   GATES_OWNER        optional; owner segment of the container name (default: the
#                      lock file's base name, e.g. cpu-build.lock -> cpu-build)
#
# Stopping a run: signal THIS script (SIGINT/SIGTERM); its exit traps remove its
# own container and scratch. Never `docker rm` by hand — a container name alone
# does not say which lane owns the run, and the rm may kill another lane's gate.
#
# The Mac arm runs by default (--mac=always): every Mac breakage so far came from
# a commit that touched no Mac-side file (a field added in sched/job.rs broke
# literals in two families' resident/ modules). --mac=auto runs it only when the
# commit touches the Mac-side set (engine/src/kernels/mlx/, the mlx-sys crate,
# engine/src/kernels/compile_key.rs, engine/src/family/*/resident/, and the
# build files engine/Cargo.toml, engine/Cargo.lock, engine/build.rs,
# engine/rust-toolchain.toml); --mac=never is for commits without Rust changes.
# The Mac arm starts first and runs concurrently with the Linux arms — it holds
# no local lock (its subshell closes the lock fd) and its rc is collected into
# the receipt at the end. It streams the archive over ssh; never a push, never
# the Mac's own checkout.
#
# Mac cleanup contract: a run removes ONLY its own scratch directory. The
# remote wrapper arms `trap 'rm -rf "$d"' EXIT` before its cd, so the remote
# shell removes its own directory on every exit of its own — its clippy's
# exit status, a remote-side timeout, or the connection's death (the shell
# runs the trap after its foreground build is done, so rm never races a live
# build). Paths where the clippy ssh never started (tar failure, connect
# failure, an interrupt in the tar phase) cannot have anything building
# remotely, so the local side removes the tree with the validated name only;
# the mac-clippy.started flag marks where the remote trap takes over. No pid
# files, no signalling of remote processes, no sweeping of other directories.
# The remote command is kept POSIX: the login shell there may not be bash.
# The local teardown (mac_teardown) is a prompt backstop for the collected
# paths.
#
# Mac disk floor: before the arm starts, free space on the volume holding
# GATES_MAC_BASE is measured with df (after mkdir -p, so a first use works);
# below GATES_MAC_MIN_FREE_GB (default 25, whole GiB) the arm refuses and the
# number goes into the receipt. One tree is about 2.2 GB; clippy needs
# headroom beyond that.
#
# Exit codes: 0 all selected arms passed; 1 unexpected command failure under
# set -e; 2 usage or environment error; 3 memory/disk guard; 4 a gate failed;
# 5 lock wait timeout; 124 a single arm exceeded GATES_ARM_MAX_SEC (reported as
# its FAIL rc, run exits 4); 129/130/143 HUP/INT/TERM during the run. A second
# run of the same hash WAITS for the shared lock up to GATES_LOCK_WAIT seconds
# and refuses with rc 5 when the wait times out.
set -euo pipefail

CTR_IMAGE=sha256:feff5d69da26e2508144318c62441fa292c0131efe91d6ccccaed4e9d1533d9d

# Remove THIS RUN's scratch directory on the Mac. The caller must have seen
# the remote command's own exit status (any cargo rc) — never a timeout (124),
# a lost connection (255) or an interrupt, where the remote may still build.
# Never signals any remote process, never touches any other directory.
mac_teardown() {
    local dir=$1
    ssh "${ssh_opts[@]}" -- "$mac_host" "rm -rf '$dir'" 2>&1 || {
        echo "NOTE: could not remove the Mac scratch dir $dir (named here for manual cleanup)" >&2
        return 0
    }
}

usage() {
    sed -n '2,57p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
}

[[ -n ${GATES_SCRATCH:-} ]] || { echo "GATES_SCRATCH is required (no /tmp default)" >&2; usage; }
commit=
mac=always
list_only=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --mac=*) mac=${1#--mac=} ;;
        --mac) shift; [[ $# -gt 0 ]] || { echo "--mac needs a value" >&2; exit 2; }; mac=$1 ;;
        --list) list_only=true ;;
        -h|--help) usage ;;
        -*) echo "unknown option: $1" >&2; usage ;;
        *) [[ -z $commit ]] || usage; commit=$1 ;;
    esac
    shift
done
[[ -n $commit ]] || usage
[[ $mac == auto || $mac == always || $mac == never ]] || usage

repo=$(git rev-parse --show-toplevel)
sha=$(git -C "$repo" rev-parse --verify "$commit^{commit}") || usage
run=$GATES_SCRATCH/$sha
logs=$run/logs
lock=${GATES_LOCK:?GATES_LOCK is required (the shared CPU-build lock path)}
# The container name carries its owner so concurrent lanes never clean up each
# other's gate containers: GATES_OWNER wins, else the lock file's base name.
owner=${GATES_OWNER:-$(basename "$lock" .lock)}
CTR_NAME="gates-${owner}-$$"
mac_max=${GATES_MAC_MAX_SEC:-1800}
arm_max=${GATES_ARM_MAX_SEC:-3600}
lock_wait=${GATES_LOCK_WAIT:-7200}
min_free=${GATES_MIN_FREE_GB:-15}
for v_name in GATES_MAC_MAX_SEC GATES_ARM_MAX_SEC GATES_LOCK_WAIT GATES_MIN_FREE_GB; do
    v_val=${!v_name:-}
    [[ -z $v_val || ($v_val =~ ^(0|[1-9][0-9]*)$ && $v_val -gt 0) ]] || { echo "$v_name must be a positive integer (no leading zeros)" >&2; exit 2; }
done
ssh_opts=(-o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=4)

if [[ $mac == auto ]]; then
    if ! git -C "$repo" rev-parse --verify -q "$sha^" >/dev/null; then
        mac=always   # no parent to diff against: run it
    else
        # A failed diff must fail the run, not select never.
        mac_side=$(git -C "$repo" diff --name-only "$sha^" "$sha") || {
            echo "GATES FAIL: git diff $sha^ $sha failed under --mac=auto" >&2
            exit 2
        }
        if grep -Eq 'engine/src/kernels/mlx/|engine/vendor/mlx-sys/|engine/src/kernels/compile_key.rs|engine/src/family/[^/]*/resident/|^engine/(Cargo\.toml|Cargo\.lock|build\.rs|rust-toolchain\.toml)$' <<<"$mac_side"; then
            mac=always
        else
            mac=never
        fi
    fi
fi

arms=(fmt dead-test-scan host-clippy host-lib published-text qual qual-validate ctr-clippy ctr-tests ctr-lib-run)
[[ $mac == never ]] || arms+=(mac-clippy)

if $list_only; then
    echo "commit  $sha"
    echo "repo    $repo"
    echo "run     $run"
    echo "lock    $lock"
    printf 'arm     %s\n' "${arms[@]}"
    exit 0
fi
if [[ $mac != never ]]; then
    [[ -n ${GATES_MAC_HOST:-} ]] || { echo "GATES_MAC_HOST is required when the Mac arm runs" >&2; exit 2; }
    mac_base=${GATES_MAC_BASE:?GATES_MAC_BASE is required when the Mac arm runs (absolute scratch base on the Mac, not /tmp)}
    [[ $mac_base == /* && $mac_base != / ]] || { echo "GATES_MAC_BASE must be absolute and not /: '$mac_base'" >&2; exit 2; }
    [[ $mac_base != *..* ]] || { echo "GATES_MAC_BASE must not contain ..: '$mac_base'" >&2; exit 2; }
    [[ $mac_base =~ ^[A-Za-z0-9_./-]+$ ]] || { echo "GATES_MAC_BASE may contain only letters, digits, _ . / - : '$mac_base'" >&2; exit 2; }
    case $mac_base in
        /tmp|/tmp/*|/var/tmp|/var/tmp/*) echo "GATES_MAC_BASE must not be under /tmp or /var/tmp: '$mac_base'" >&2; exit 2 ;;
    esac
fi
mac_host=${GATES_MAC_HOST:-none}
mac_base=${GATES_MAC_BASE:-none}
echo "$(date -u +%FT%TZ) waiting up to ${lock_wait}s for the CPU-build lock ($lock) for $sha" >&2
exec 9>"$lock"
if ! flock -w "$lock_wait" 9; then
    echo "GATES FAIL: refused, lock timeout after ${lock_wait}s waiting for $lock (another run holds it)" >&2
    exit 5
fi
echo "$(date -u +%FT%TZ) lock acquired for $sha" >&2
available=$(awk '$1 == "MemAvailable:" {print $2; exit}' /proc/meminfo)
[[ -n $available && $available -ge 12582912 ]] || { echo "need >= 12G MemAvailable, have ${available:-none} kB" >&2; exit 3; }
mkdir -p "$GATES_SCRATCH"   # a first use must reach the disk guard, not abort at df
free_g=$(df -Bg --output=avail "$(dirname "$run")" | tail -1 | tr -dc '0-9')
[[ $free_g -ge $min_free ]] || { echo "need >= ${min_free}G free on the scratch filesystem, have ${free_g}G" >&2; exit 3; }
mkdir -p "$logs" "$run/tmp"

start_ts=$(date -u +%FT%TZ)
receipt=("EXACT-TIP GATES $start_ts commit=$sha platform=$(uname -sm) mac=$mac"
         "container=$CTR_NAME script_pid=$$")
arm_status=()
mac_job=
mac_dir=

finish() {
    rc=$?
    trap '' INT TERM HUP   # cleanup itself must not be interrupted
    if [[ -n $mac_job ]]; then
        # Interrupted run: kill the LOCAL driver subshell (its ssh child dies
        # with it). If the clippy ssh had started, the remote shell's own EXIT
        # trap removes its directory when its build ends. If it had not (the
        # interrupt landed in the tar phase), nothing on the Mac can still be
        # building, so the local side removes the tree (validated name only).
        pkill -TERM -P "$mac_job" 2>/dev/null || true
        kill "$mac_job" 2>/dev/null || true
        wait "$mac_job" 2>/dev/null || true
        mac_job=
        if [[ -f $logs/mac-clippy.started ]]; then
            receipt+=("NOTE: interrupted with the Mac arm live; the remote shell's EXIT trap removes $mac_dir when its build ends")
        else
            local_dir=$mac_dir
            mac_dir=
            mac_teardown "$local_dir"
            receipt+=("NOTE: interrupted before the Mac clippy started; removed $local_dir locally")
        fi
    elif [[ -n $mac_dir && $mac_dir == "$mac_base"/gates.[A-Za-z0-9][A-Za-z0-9][A-Za-z0-9][A-Za-z0-9][A-Za-z0-9][A-Za-z0-9][A-Za-z0-9][A-Za-z0-9][A-Za-z0-9][A-Za-z0-9] ]]; then
        # The Mac arm already returned its own exit status and its directory
        # was NOT removed in the collection block (that block only removes on
        # a real cargo status): leave it alone here too. Nothing to do.
        :
    fi
    docker rm -f "$CTR_NAME" >/dev/null 2>&1 || true
    git -C "$repo" worktree remove --force "$run/src" 2>/dev/null || rm -rf "$run/src"
    rm -rf "$run/target-host" "$run/target-ctr" "$run/tmp"
    if [[ ! -f $logs/receipt.txt ]]; then
        receipt+=("${arm_status[@]}")
        receipt+=("ended $(date -u +%FT%TZ) ABORTED rc=$rc")
        printf '%s\n' "${receipt[@]}" > "$logs/receipt.txt"
    fi
    if [[ $rc -ne 0 ]]; then
        echo "GATES FAIL rc=$rc commit=$sha logs=$logs (build trees deleted)" >&2
    fi
    exit "$rc"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP


# Clear this run's own stale state from a kill -9 — the worktree registration
# is removed by path (never a repository-wide prune), together with target dirs,
# tmp and any stale per-arm exit files.
git -C "$repo" worktree remove --force "$run/src" 2>/dev/null || true
rm -rf "$run/src" "$run/target-host" "$run/target-ctr"
wt_list=$(git -C "$repo" worktree list --porcelain)
if grep -qx "worktree $run/src" <<<"$wt_list"; then
    echo "GATES FAIL: '$run/src' is still a registered worktree whose directory is gone;" >&2
    echo "remove the registration first: git -C \"$repo\" worktree remove --force \"$run/src\"" >&2
    exit 2
fi
mkdir -p "$run/tmp"

rm -f "$logs"/*.exit "$logs/receipt.txt" "$logs/mac-clippy.started"
git -C "$repo" worktree add --detach --quiet "$run/src" "$sha"

export TMPDIR=$run/tmp TMP=$run/tmp TEMP=$run/tmp
export CARGO_TARGET_DIR=$run/target-host CARGO_BUILD_JOBS=4 CARGO_NET_OFFLINE=true
export CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0 CARGO_TERM_COLOR=never CARGO_INCREMENTAL=0
export PATH="${CARGO_HOME:-$HOME/.cargo}/bin:$PATH"

run_arm() {
    local label=$1 duration
    shift
    local started=$SECONDS
    if timeout -k 30 "$arm_max" "$@" > "$logs/$label.log" 2>&1; then
        printf '%s\n' 0 > "$logs/$label.exit"
    else
        local rc=$?
        printf '%s\n' "$rc" > "$logs/$label.exit"
        arm_status+=("FAIL $label rc=$rc log=$logs/$label.log")
        echo "FAIL $label rc=$rc log=$logs/$label.log" >&2
        return "$rc"
    fi
    duration=$((SECONDS - started))
    arm_status+=("PASS $label ${duration}s")
    printf 'PASS %-14s %3ds %s\n' "$label" "$duration" "$logs/$label.log"
}

# Everything after <after> in the master arm list is still pending when the arm
# named <failed> stops the run; the Mac, which ran concurrently, is NOT-COLLECTED
# unless it never started (a mac-setup failure lists it NOT-RUN). An empty
# <after> (failure before the first arm) lists every arm as NOT-RUN.
fail_now() {
    local failed=$1 after=$2 i label
    local start=-1 mac_ran=true
    for i in "${!arms[@]}"; do
        [[ ${arms[i]} == "$after" ]] && { start=$i; break; }
    done
    [[ $failed == mac-setup ]] && mac_ran=false
    for ((i = start + 1; i < ${#arms[@]}; i++)); do
        label=${arms[i]}
        if [[ $label == mac-clippy && $mac_ran == true ]]; then
            arm_status+=("NOT-COLLECTED mac-clippy (concurrent; the remote shell's EXIT trap removes its dir; log $logs/mac-clippy.driver.log)")
        else
            arm_status+=("NOT-RUN $label (stopped at $failed)")
        fi
    done
    receipt+=("${arm_status[@]}")
    receipt+=("ended $(date -u +%FT%TZ) FAIL at $failed")
    printf '%s\n' "${receipt[@]}" | tee "$logs/receipt.txt"
    exit 4
}

# The qualification tool runs from the repo root; these tiny wrappers let the
# per-arm timeout apply to it like any external command.
cat > "$run/tmp/qual-generate.sh" <<QUAL
cd "$run/src" && exec python3 scripts/qual_inventory.py generate
QUAL
cat > "$run/tmp/qual-validate.sh" <<QUAL
cd "$run/src" && exec python3 scripts/qual_inventory.py validate
QUAL

cd "$run/src/engine"

# The Mac arm starts first and runs concurrently with the Linux arms: its
# subshell closes the lock fd (fd 9) so its ssh children never hold the local
# lock, and its rc is collected into the receipt at the end. The remote shell
# owns its cleanup: its EXIT trap removes its scratch directory on every exit
# of its own, so no local path can leave a tree behind.
if [[ $mac != never ]]; then
    # Disk floor before anything is created remotely (F5): one tree is about
    # 2.2 GB and clippy needs headroom; 63 left-behind trees filled the Mac's
    # data volume on 2026-11-17. mkdir first so a first use of the base works;
    # df -g: 1 GiB blocks, column 4 = available.
    mac_min_free_gb=${GATES_MAC_MIN_FREE_GB:-25}
    [[ $mac_min_free_gb =~ ^[0-9]+$ ]] || {
        echo "GATES_MAC_MIN_FREE_GB must be a whole number of GiB: '$mac_min_free_gb'" >&2
        exit 2
    }
    # 10# rejects octal reading: "08" would abort the arithmetic below.
    mac_min_free_gb=$((10#$mac_min_free_gb))
    if ! mac_free_gb=$(ssh "${ssh_opts[@]}" -- "$mac_host" "mkdir -p '$mac_base' && df -g '$mac_base' | awk 'NR==2 {print \$4; exit}'"); then
        arm_status+=("FAIL mac-clippy rc=unreachable-host log=$logs/mac-clippy.log")
        fail_now mac-setup ''
    fi
    [[ $mac_free_gb =~ ^[0-9]+$ ]] || {
        echo "GATES FAIL: the Mac returned an unexpected df value: '$mac_free_gb'" >&2
        arm_status+=("FAIL mac-clippy rc=bad-df-output log=$logs/mac-clippy.log")
        fail_now mac-setup ''
    }
    if (( 10#$mac_free_gb < mac_min_free_gb )); then
        echo "GATES FAIL: Mac free space ${mac_free_gb} GiB below the ${mac_min_free_gb} GiB floor on $mac_base" >&2
        arm_status+=("FAIL mac-clippy rc=mac-low-disk log=$logs/mac-clippy.log (free ${mac_free_gb} GiB < floor ${mac_min_free_gb} GiB)")
        fail_now mac-setup ''
    fi
    arm_status+=("mac-disk free=${mac_free_gb}GiB floor=${mac_min_free_gb}GiB")
    if ! mac_dir=$(ssh "${ssh_opts[@]}" -- "$mac_host" "mktemp -d '$mac_base/gates.XXXXXXXXXX'"); then
        arm_status+=("FAIL mac-clippy rc=unreachable-host log=$logs/mac-clippy.log")
        fail_now mac-setup ''
    fi
    # mktemp's own output is the only accepted shape — exactly gates. plus the
    # ten characters mktemp draws from — before it is ever used in a command.
    # A rejected name is logged and NEVER passed to rm -rf: it can be empty,
    # the base dir, or anything else, so nothing is removed on this path.
    [[ $mac_dir == "$mac_base"/gates.[A-Za-z0-9][A-Za-z0-9][A-Za-z0-9][A-Za-z0-9][A-Za-z0-9][A-Za-z0-9][A-Za-z0-9][A-Za-z0-9][A-Za-z0-9][A-Za-z0-9] ]] || {
        echo "GATES FAIL: the Mac returned an unexpected scratch dir name: '$mac_dir'" >&2
        arm_status+=("FAIL mac-clippy rc=bad-scratch-name (logged, nothing removed)")
        fail_now mac-setup ''
    }
    (
        exec 9>&-   # the lock fd must not leak into the ssh children
        # The trap arms before cd, so even a failed cd still removes the tree.
        # Keep the command POSIX: the remote login shell may not be bash.
        remote_cmd="d='$mac_dir'; trap 'rm -rf \"\$d\"' EXIT; cd \"\$d/engine\" && PATH=/opt/homebrew/bin:\$HOME/.cargo/bin:\$PATH CARGO_TARGET_DIR=\"\$d/target\" cargo clippy --all-targets --locked --features native-mlx -- -D warnings"
        if ! git -C "$repo" archive --format=tar "$sha" |
            ssh "${ssh_opts[@]}" -- "$mac_host" "tar -x -C '$mac_dir'"; then
            # The clippy ssh never started, so nothing on the Mac can still be
            # building a clippy; the local side removes the tree (validated
            # name only). A remote tar that is still draining its EOF exits
            # on its own - this rm may race it, which is accepted: the tree is
            # this run's own and no result is read from it afterwards.
            mac_teardown "$mac_dir"
            exit 1
        fi
        touch "$logs/mac-clippy.started"
        run_arm mac-clippy timeout -k 30 "$mac_max" ssh "${ssh_opts[@]}" -- "$mac_host" "$remote_cmd"
    ) > "$logs/mac-clippy.driver.log" 2>&1 &
    mac_job=$!
fi

run_arm fmt cargo fmt --all --check || fail_now fmt fmt
run_arm dead-test-scan python3 "$run/src/scripts/dead_test_scan.py" || fail_now dead-test-scan dead-test-scan
run_arm host-clippy cargo clippy --locked --all-targets -- -D warnings || fail_now host-clippy host-clippy
run_arm host-lib cargo test --locked --lib || fail_now host-lib host-lib
run_arm published-text cargo test --locked --test published_text || fail_now published-text published-text

# The qualification gate: the committed file must be exactly what the source
# regenerates (drift FAILS the run), and validation reads the committed file.
facts="$run/src/scripts/qual_facts.json"
facts_before=$(md5sum "$facts")
run_arm qual bash "$run/tmp/qual-generate.sh" || fail_now qual qual
if [[ $(md5sum "$facts") != "$facts_before" ]]; then
    arm_status+=("FAIL qual-drift rc=1 scripts/qual_facts.json differs from the committed file")
    echo "FAIL qual-drift: scripts/qual_facts.json differs from the committed file" >&2
    git -C "$run/src" checkout -- scripts/qual_facts.json
    fail_now qual-drift qual
fi
git -C "$run/src" checkout -- scripts/qual_facts.json
run_arm qual-validate bash "$run/tmp/qual-validate.sh" || fail_now qual-validate qual-validate

ctr=(docker run --rm --pull=never --name "$CTR_NAME" --label fifi.owner=exact-tip-gates
    --user "$(id -u):$(id -g)" --network none --memory 24g --memory-swap 24g --cpus 4
    -v "$run/src:$run/src" -v "$run/target-ctr:$run/target-ctr" -v "$run/tmp:$run/tmp"
    -v "${CARGO_HOME:-$HOME/.cargo}:/host-cargo:ro" -v "${RUSTUP_HOME:-$HOME/.rustup}:/host-rustup:ro"
    -e CARGO_HOME=/host-cargo -e RUSTUP_HOME=/host-rustup -e "CARGO_TARGET_DIR=$run/target-ctr"
    -e CARGO_BUILD_JOBS=4 -e CARGO_NET_OFFLINE=true -e CARGO_PROFILE_RELEASE_DEBUG=0
    -e CARGO_TERM_COLOR=never -e CARGO_INCREMENTAL=0 -e CUDA_VISIBLE_DEVICES= -e NVIDIA_VISIBLE_DEVICES=void
    -e "HOME=$run/tmp" -e "TMPDIR=$run/tmp" -e "TMP=$run/tmp" -e "TEMP=$run/tmp" -e CUDA_PATH=/usr/local/cuda-12.6
    -e PATH=/host-cargo/bin:/usr/local/cuda-12.6/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
    -w "$run/src/engine" "$CTR_IMAGE" cargo)
# Docker creates a missing bind-mount path as root; the container runs as the
# invoking uid, so both local target dirs exist before the run.
mkdir -p "$run/target-host" "$run/target-ctr"
export CARGO_TARGET_DIR=$run/target-ctr
run_arm ctr-clippy "${ctr[@]}" clippy --release --locked --all-targets --features native-cuda -- -D warnings || fail_now ctr-clippy ctr-clippy
run_arm ctr-tests "${ctr[@]}" test --release --locked --tests --no-run --features native-cuda || fail_now ctr-tests ctr-tests
# ctr-tests only BUILDS the native-cuda tests (--no-run), so the device-free tests under
# cfg(feature = "native-cuda") never executed in any battery. Run the lib test executable in
# the same no-GPU container (build reused, no recompile). No --include-ignored: ignored device
# tests stay out; non-ignored device tests skip via testkit::device_present. --nocapture so the
# SKIPPED lines are countable. The container mounts $run/src at the compile-time
# CARGO_MANIFEST_DIR path, so source-tree tests (wrapper_completeness) find their files.
run_arm ctr-lib-run "${ctr[@]}" test --release --locked --lib --features native-cuda -- --test-threads=1 --nocapture || fail_now ctr-lib-run ctr-lib-run

# Collect the concurrent Mac arm into the receipt. The subshell's own PASS/FAIL
# line went to its driver log; the exit file it wrote carries the rc here.
# Collected statuses (including every real cargo rc) are removed promptly by
# mac_teardown; a local timeout kill (124 or 125), a lost connection (255) or
# no exit file at all leaves removal to the remote shell's own EXIT trap.
if [[ -n $mac_job ]]; then
    mac_rc=$(cat "$logs/mac-clippy.exit" 2>/dev/null || echo unknown)
    if wait "$mac_job" && [[ $mac_rc != unknown ]]; then
        arm_status+=("PASS mac-clippy ($mac_rc) $logs/mac-clippy.log")
        local_dir=$mac_dir
        mac_job=; mac_dir=
        mac_teardown "$local_dir"
    else
        arm_status+=("FAIL mac-clippy rc=$mac_rc log=$logs/mac-clippy.log")
        mac_job=      # the driver subshell has exited; finish() must not kill again
        if [[ $mac_rc == 124 || $mac_rc == 125 || $mac_rc == 255 || $mac_rc == unknown ]]; then
            if [[ $mac_rc == 255 ]]; then
                # A lost connection gives no evidence the remote trap armed
                # (the ssh may have failed before the remote shell ran), so the
                # local side removes the tree. If the trap did arm, the remote
                # rm and this one are both idempotent.
                local_dir=$mac_dir
                mac_dir=
                mac_teardown "$local_dir"
                arm_status+=("mac-clippy: connection lost; removed $local_dir locally")
            elif [[ -f $logs/mac-clippy.started ]]; then
                arm_status+=("mac-clippy: removal left to the remote EXIT trap (remote may still be building): $mac_dir")
                mac_dir=
            else
                local_dir=$mac_dir
                mac_dir=
                mac_teardown "$local_dir"
                arm_status+=("mac-clippy: the clippy ssh never started; removed $local_dir locally")
            fi
        else
            local_dir=$mac_dir
            mac_dir=
            mac_teardown "$local_dir"
        fi
        fail_now mac-clippy mac-clippy
    fi
fi

exe_count=$(grep -c 'Executable' "$logs/ctr-tests.log" || true)
# Report the lib-run result plus the device-present SKIPPED count, and refuse a
# run that executed too few tests: a run where nearly everything skipped must
# not read as a pass of the device-free tests (executed = passed - skipped).
lib_line=$(grep 'test result:' "$logs/ctr-lib-run.log" | tail -1)
lib_skipped=$(grep -c 'SKIPPED .*: no CUDA device' "$logs/ctr-lib-run.log" || true)
lib_passed=$(grep -oE '[0-9]+ passed' <<<"$lib_line" | head -1 | awk '{print $1}')
lib_executed=$(( ${lib_passed:-0} - lib_skipped ))
# Executed floor: the first green no-GPU lib run (8c7d6a4f6, 2026-11-17) executed
# 4938 tests; the floor is that number minus ~7% rounded down, so a mass-skip
# regression (every device test skipping and nothing executing) cannot read
# as green. Raise it when the suite grows materially.
lib_floor=4600
if (( lib_executed < lib_floor )); then
    # Replace run_arm's PASS line so the receipt carries exactly one
    # ctr-lib-run status.
    rebuilt=()
    for status_line in "${arm_status[@]}"; do
        [[ $status_line == "PASS ctr-lib-run "* ]] || rebuilt+=("$status_line")
    done
    arm_status=("${rebuilt[@]}")
    arm_status+=("FAIL ctr-lib-run (only ${lib_executed} executed, below the ${lib_floor}-test floor)")
    fail_now ctr-lib-run ctr-lib-run
fi
receipt+=("${arm_status[@]}")
receipt+=("ctr-tests ${exe_count:-0} release test executables built")
receipt+=("ctr-lib-run ${lib_line#test result: }")
receipt+=("ctr-lib-run ${lib_skipped} printed SKIPPED (no CUDA device); ${lib_executed} executed (passed minus skipped)")
receipt+=("started $start_ts")
receipt+=("ended $(date -u +%FT%TZ) OK")
echo
echo '--- exact-tip-gates receipt ---'
printf '%s\n' "${receipt[@]}" | tee "$logs/receipt.txt"
echo "scratch $logs (build trees deleted)"
