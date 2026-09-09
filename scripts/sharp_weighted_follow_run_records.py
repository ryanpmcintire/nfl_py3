"""Execute the saved weak-signals record argv lists one at a time against the main registry."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

DATA_REPO = Path("F:/Repos/nfl_py3")
COMMANDS = DATA_REPO / "artifacts/sharp_weighted_follow/20260909T233606Z/record_commands.json"
EXE = DATA_REPO / ".venv/Scripts/nfl-ats.exe"


def main() -> None:
    env = dict(os.environ)
    env["NFL_ATS_REGISTRY_DIR"] = str(DATA_REPO / "registry")
    commands = json.loads(COMMANDS.read_text(encoding="utf-8"))
    failures = 0
    for argv in commands:
        proc = subprocess.run(
            [str(EXE), *argv], cwd=str(DATA_REPO), env=env, capture_output=True, text=True
        )
        status = "OK " if proc.returncode == 0 else "FAIL"
        if proc.returncode != 0:
            failures += 1
        tail = (proc.stdout.strip() or proc.stderr.strip()).splitlines()
        print(f"{status} {argv[3]} :: {tail[-1] if tail else ''}")
    print(f"{len(commands) - failures} recorded, {failures} failed")


if __name__ == "__main__":
    main()
