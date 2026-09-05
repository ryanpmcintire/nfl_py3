"""Full verification tier: the release gate (ENG-11).

Why this exists
----------------
This is the unchanged set of commands `AGENTS.md` "Required verification"
already names as release-blocking -- ``ruff format --check .``,
``ruff check .``, ``mypy src``, ``pytest`` (the whole suite, no marker
filter) -- run back to back with per-step wall time printed. Nothing here is
new policy: ``scripts/verify_fast.py`` is the new artifact (a faster subset
for the PR loop, via ``-m "not full"``); this script exists so the full,
unabridged gate has a single command too, and so the capture scheduler
(``scripts/capture_scheduler.py``) has something schedulable for a periodic
run. See ``docs/verification_tiers.md`` for what each tier covers and the
measured wall time of each, and AGENTS.md's "Automatic session handoff"
section for what must additionally happen before a push to master
(``nfl-ats handoff`` / ``nfl-ats handoff --check``) -- this script does not
refresh the handoff; it only re-verifies safety/typing/lint/tests.

Determinism / replay checks
----------------------------
ENG-11's definition of done also asks for "the existing determinism/replay
checks if any script exists". As of 2026-09-04 there is no single
general-purpose, repo-wide determinism or run-replay script to add here:
``scripts/lockday_package_verify.py`` re-hashes one already-written lock-day
decision package (it takes a package path and needs a package to exist --
not something this gate can run unconditionally every time), and the
dedicated reproducible-run-replay command described in ROADMAP.md's ENG-13
("Reproducible run replay") has not been built yet. The full ``pytest`` run
below already includes this repo's per-feature determinism/reproduction
tests (e.g. ``tests/test_cfb_audit.py``, ``tests/test_postseason.py``'s
bit-identical check, ``tests/test_dependence.py``'s determinism check --
several of these are exactly the tests ``scripts/verify_fast.py`` deselects
via the ``full`` marker). When ENG-13 ships a real replay command, add it as
a fifth step here.

Usage
-----
    .tools\\uv.exe run python scripts/verify_full.py

Exit code is 0 only when every step passes. Intended to be run before every
push to master (in addition to, not instead of, ``nfl-ats handoff --check``)
and from the capture scheduler's disabled-by-default weekly ``verify_full``
job (SCHEDULE in ``scripts/capture_scheduler.py``) -- enable that job
deliberately when a periodic unattended run is wanted; it is not enabled by
default because a multi-minute pytest run is not something to fire
unattended without a decision to do so.
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
# directory (a pytest INTERNALERROR, not a real test failure). This is the
# one deliberate addition to the otherwise-unchanged AGENTS.md pytest
# invocation below: it changes WHERE pytest writes temp files, not what runs
# or what counts as pass/fail.
_BASETEMP = Path(tempfile.gettempdir()) / f"nfl_ats_verify_full_{os.getpid()}"

# The four AGENTS.md "Required verification" gates, unchanged -- same
# commands, same order, no --no-sync (this is the release gate, not the fast
# PR loop; it should always run against a freshly-synced environment).
STEPS: list[tuple[str, list[str]]] = [
    ("ruff format --check .", [str(UV), "run", "ruff", "format", "--check", "."]),
    ("ruff check .", [str(UV), "run", "ruff", "check", "."]),
    ("mypy src", [str(UV), "run", "mypy", "src"]),
    (
        "pytest (full suite)",
        [str(UV), "run", "pytest", "--basetemp", str(_BASETEMP)],
    ),
]


def main() -> int:
    if not UV.is_file():
        print(f"error: uv not found at {UV}", file=sys.stderr)
        return 1

    results: list[tuple[str, bool, float]] = []
    overall_start = time.monotonic()
    for name, cmd in STEPS:
        print(f"\n=== full: {name} ===", flush=True)
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
    print("\n=== full tier (release gate) summary ===")
    for name, ok, elapsed in results:
        print(f"{'PASS' if ok else 'FAIL':<4} {elapsed:7.1f}s  {name}")
    ran_names = {name for name, _, _ in results}
    for name, _ in STEPS:
        if name not in ran_names:
            print(f"SKIP     -.-s  {name}")
    print(f"total: {total_elapsed:.1f}s")

    if not all(ok for _, ok, _ in results) or len(results) < len(STEPS):
        print("\nFULL TIER (RELEASE GATE) FAILED.", file=sys.stderr)
        return 1

    print(
        "\nFull tier passed. This IS the AGENTS.md release gate; also run "
        "`nfl-ats handoff --check` before pushing master, per AGENTS.md "
        "'Automatic session handoff'."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
