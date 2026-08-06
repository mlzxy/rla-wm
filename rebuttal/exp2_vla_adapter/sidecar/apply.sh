#!/usr/bin/env bash
# NOTE: these patches document the sidecar integration as it was written, against commit 4e05436
# of the development repo. They are kept as a record of the diff, not as a step you have to run.
# The runnable path -- extract once, then read the sidecar -- is in this directory's README.md.
#
# Apply the RLA -> VLA-Adapter session bundle. Run from the repo root.
#
#   apply.sh --check              dry run of 00-all.patch, touches nothing
#   apply.sh                      apply 00-all.patch
#   apply.sh --split              apply 01..07 in order, stop at the first failure
#   apply.sh --split --only 03    apply just one piece
#
# --check works with every combination above.
set -euo pipefail

BASE_COMMIT=4e05436
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCHES="$HERE/patches"

check=0 split=0 only=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --check)   check=1 ;;
        --split)   split=1 ;;
        --only)    only="${2:?--only needs a patch number, e.g. --only 03}"; split=1; shift ;;
        -h|--help) sed -n '6,13p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)         echo "unknown flag: $1" >&2; exit 2 ;;
    esac
    shift
done

# --- sanity ----------------------------------------------------------------
git rev-parse --show-toplevel >/dev/null 2>&1 || { echo "not a git repo" >&2; exit 1; }
root="$(git rev-parse --show-toplevel)"
[[ "$PWD" == "$root" ]] || { echo "run from the repo root: cd $root" >&2; exit 1; }

head_short="$(git rev-parse --short HEAD)"
if [[ "$head_short" != "$BASE_COMMIT"* ]]; then
    echo "note: HEAD is $head_short, bundle was cut against $BASE_COMMIT" >&2
fi

# Plain `git apply` first -- it leaves the index alone. Fall back to --3way only when the context
# has drifted; that mode does stage whatever it merges.
try_apply() {  # try_apply <patch> [extra git-apply flags...]  -> 0 clean / 3 needed --3way / 1 no
    local p="$1"; shift
    git apply "$@" "$p" 2>/dev/null && return 0
    git apply "$@" --3way "$p" 2>/dev/null && return 3
    return 1
}

# --- pick the patch list ---------------------------------------------------
if (( split )); then
    if [[ -n "$only" ]]; then
        mapfile -t list < <(ls "$PATCHES/${only}"-*.patch 2>/dev/null || true)
        (( ${#list[@]} )) || { echo "no patch numbered '$only' in $PATCHES" >&2; exit 1; }
    else
        mapfile -t list < <(ls "$PATCHES"/0[1-7]-*.patch)
    fi
else
    list=("$PATCHES/00-all.patch")
fi

# --- go --------------------------------------------------------------------
mode="applying"; (( check )) && mode="checking"
echo "$mode ${#list[@]} patch(es) against $head_short"

failed=0
for p in "${list[@]}"; do
    name="$(basename "$p")"
    if (( check )); then
        rc=0; try_apply "$p" --check || rc=$?   # `|| rc=$?` so `set -e` does not fire on 1 or 3
        case $rc in
            0) printf '  OK      %s\n' "$name" ;;
            3) printf '  OK/3way %s (context drifted, will stage what it merges)\n' "$name" ;;
            *) printf '  FAIL    %s\n' "$name"
               git apply --check "$p" 2>&1 | sed 's/^/          /' || true
               failed=1 ;;
        esac
    else
        rc=0; try_apply "$p" || rc=$?
        case $rc in
            0) printf '  applied      %s\n' "$name" ;;
            3) printf '  applied/3way %s\n' "$name" ;;
            *) printf '  FAILED       %s -- stopping\n' "$name" >&2
               git apply "$p" 2>&1 | sed 's/^/          /' >&2 || true
               echo "  fall back to: rsync -a $HERE/new-files/ ./   (new files only, never conflicts)" >&2
               exit 1 ;;
        esac
    fi
done

if (( check )); then
    (( failed )) && { echo "some patches do not apply cleanly; try --split to isolate" >&2; exit 1; }
    echo "all clean -- re-run without --check to apply"
else
    echo "done. next: git status, then docs/rla-vla-adapter-runbook.md"
fi
