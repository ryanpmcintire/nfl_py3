"""Fast PR-speed verification tier (ENG-11).

Why this exists
----------------
`AGENTS.md` "Required verification" names four gates -- ``ruff format
--check``, ``ruff check``, ``mypy src``, ``pytest`` -- and the full ``pytest``
run now covers 3,700+ tests (measured 2026-09-04:
``pytest --collect-only`` -> 3,612 tests at the start of that session, 3,729
by its end as concurrent agents added files). Most of that time is not typing
or lint feedback; it is a small number of tests that fit a real model, read
real on-disk data/artifacts, or reproduce a full build bit-for-bit
(``pytest --durations=40``, same session: top item 12.0s, next three 6-10s,
each individually far above the sub-100ms cost of a typical unit test).

This script is the FAST tier: safety, typing, lint, and the REST of the test
suite (``-m "not full"``), so a PR check finishes quickly without dropping
coverage of anything AGENTS.md calls release-blocking. It is deliberately NOT
a substitute for the release gate -- see ``docs/verification_tiers.md`` and
``scripts/verify_full.py``. The tests deselected by ``-m "not full"`` are
tagged in the test files themselves (search for ``ENG-11`` in ``tests/``);
none of them are safety, leakage, or point-in-time-chronology tests -- those
stay in this fast tier deliberately, per AGENTS.md's "Required verification"
and this project's leakage-regression-test invariant.

Usage
-----
    .tools\\uv.exe run --no-sync python scripts/verify_fast.py

Exit code is 0 only when every step passes. Uses ``--no-sync`` (skips the
lockfile re-resolution ``uv run`` would otherwise do on every invocation) for
PR-loop speed; ``scripts/verify_full.py`` runs the AGENTS.md commands
unchanged, without ``--no-sync``, as the release gate.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
UV = REPO / ".tools" / "uv.exe"
# pytest's default basetemp (a numbered dir under the OS temp dir, shared per
# OS user) collides with any other pytest process running concurrently under
# the same account -- measured 2026-09-04 on this repo's multi-agent fleet
# sessions: a concurrent run's numbered-dir cleanup hit
# `PermissionError: Access is denied` scanning another run's still-open
# directory (pytest INTERNALERROR, not a real test failure). Give this run
# its own directory, keyed by PID, instead.
_BASETEMP = Path(tempfile.gettempdir()) / f"nfl_ats_verify_fast_{os.getpid()}"

STEPS: list[tuple[str, list[str]]] = [
    ("ruff format --check .", [str(UV), "run", "--no-sync", "ruff", "format", "--check", "."]),
    ("ruff check .", [str(UV), "run", "--no-sync", "ruff", "check", "."]),
    ("mypy src", [str(UV), "run", "--no-sync", "mypy", "src"]),
    (
        'pytest -m "not full"',
        [
            str(UV),
            "run",
            "--no-sync",
            "pytest",
            "-m",
            "not full",
            "--basetemp",
            str(_BASETEMP),
        ],
    ),
]


def main() -> int:
    if not UV.is_file():
        print(f"error: uv not found at {UV}", file=sys.stderr)
        return 1

    results: list[tuple[str, bool, float]] = []
    overall_start = time.monotonic()
    for name, cmd in STEPS:
        print(f"\n=== fast: {name} ===", flush=True)
        step_start = time.monotonic()
        completed = subprocess.run(cmd, cwd=REPO)
        elapsed = time.monotonic() - step_start
        ok = completed.returncode == 0
        results.append((name, ok, elapsed))
        status = "PASS" if ok else "FAIL"
        print(f"--- {status} ({elapsed:.1f}s): {name} ---", flush=True)
        if not ok:
            break

    total_elapsed = time.monotonic() - overall_start
    print("\n=== fast tier summary ===")
    for name, ok, elapsed in results:
        print(f"{'PASS' if ok else 'FAIL':<4} {elapsed:7.1f}s  {name}")
    ran_names = {name for name, _, _ in results}
    for name, _ in STEPS:
        if name not in ran_names:
            print(f"SKIP     -.-s  {name}")
    print(f"total: {total_elapsed:.1f}s")

    if not all(ok for _, ok, _ in results) or len(results) < len(STEPS):
        print(
            "\nFAST TIER FAILED. This tier is NOT the release gate -- a pass here is "
            "necessary but not sufficient for a push to master. Run "
            "scripts/verify_full.py before pushing master (AGENTS.md "
            "'Required verification').",
            file=sys.stderr,
        )
        return 1

    print(
        "\nFast tier passed. Reminder: this is NOT sufficient for a master push -- "
        "run scripts/verify_full.py first (docs/verification_tiers.md)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
