from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
UV = REPO / ".tools" / "uv.exe"
_BASETEMP = Path(tempfile.gettempdir()) / f"nfl_ats_verify_full_{os.getpid()}"

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
