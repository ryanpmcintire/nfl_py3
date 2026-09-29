from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
UV = REPO / ".tools" / ("uv.exe" if sys.platform == "win32" else "uv")
SITE_PATHS = (
    "docs/index.html",
    "docs/model.html",
    "docs/history.html",
    "docs/findings.html",
    "docs/.nojekyll",
    "CURRENT_PREDICTIONS.md",
    "tiebreaker.json",
)
CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


def _run(args: list[str], *, timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=REPO,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        creationflags=CREATE_NO_WINDOW,
    )


def _git(*args: str, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return _run(["git", *args], timeout=timeout)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild the public site from current artifacts, then commit and push only the "
            "site files so GitHub Pages shows settled results without a manual session."
        )
    )
    parser.add_argument("--dry", action="store_true", help="Rebuild and report; no commit or push.")
    args = parser.parse_args(argv)

    branch = _git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    if branch != "master":
        print(json.dumps({"status": "skipped", "reason": f"checked out on {branch!r}"}))
        return 1

    board = _run([str(UV), "run", "--no-sync", "nfl-ats", "publish-board"], timeout=1200)
    if board.returncode != 0:
        tail = (board.stderr or board.stdout).strip().splitlines()[-5:]
        print(json.dumps({"status": "failed", "step": "publish-board", "error": tail}))
        return 1

    tracked = [path for path in SITE_PATHS if (REPO / path).exists()]
    changed = _git("status", "--porcelain", "--", *tracked).stdout.strip()
    if not changed:
        print(json.dumps({"status": "unchanged"}))
        return 0
    if args.dry:
        print(json.dumps({"status": "dry_run", "changed": changed.splitlines()}))
        return 0

    commit = _git(
        "commit",
        "-m",
        "Republish the site with the latest settled results",
        "--",
        *tracked,
    )
    if commit.returncode != 0:
        print(json.dumps({"status": "failed", "step": "commit", "error": commit.stderr.strip()}))
        return 1
    push = _git("push", "origin", "master", timeout=300)
    if push.returncode != 0:
        print(json.dumps({"status": "failed", "step": "push", "error": push.stderr.strip()}))
        return 1
    head = _git("rev-parse", "--short", "HEAD").stdout.strip()
    print(json.dumps({"status": "published", "commit": head, "changed": changed.splitlines()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
