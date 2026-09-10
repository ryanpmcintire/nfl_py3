from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
UV = REPO / ".tools" / "uv.exe"
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
