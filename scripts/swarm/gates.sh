#!/usr/bin/env bash
# Run the four quality gates against a worktree using the main repo's venv.
# Usage: gates.sh <worktree-path> [basetemp]
set -uo pipefail
WT=$1
BT=${2:-C:/Users/Ryan/AppData/Local/Temp/nflats-gates-$RANDOM}
VENV=/f/Repos/nfl_py3/.venv/Scripts
fail=0

cd "$WT" || exit 1
echo "== ruff format =="
"$VENV/ruff.exe" format --check . || fail=1
echo "== ruff check =="
"$VENV/ruff.exe" check . || fail=1
echo "== mypy =="
"$VENV/mypy.exe" src || fail=1
echo "== pytest =="
PYTHONPATH="$(cygpath -m "$WT")/src;${PYTHONPATH:-}" \
  "$VENV/python.exe" -m pytest -q --basetemp="$BT" || fail=1

if [ "$fail" -eq 0 ]; then echo "GATES_PASS"; else echo "GATES_FAIL"; exit 1; fi
