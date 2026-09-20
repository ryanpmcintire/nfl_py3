from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from nfl_ats.active_model import load_active_ats_model
from nfl_ats.public_board import active_artifact_path

REPO = Path(__file__).resolve().parents[1]
UV = REPO / ".tools" / "uv.exe"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Refresh the current week's lineup forecast")
    parser.add_argument(
        "--dry-run", action="store_true", help="show the weekly plan without writing"
    )
    args = parser.parse_args(argv)
    artifacts = REPO / "artifacts"
    active = load_active_ats_model(artifacts)
    if active is None:
        raise SystemExit("No synchronized active model is available")
    forecast = active_artifact_path(artifacts, active, "weekly_forecast")
    if forecast is None:
        raise SystemExit("Active model has no linked weekly forecast")
    metadata = json.loads((forecast / "metadata.json").read_text(encoding="utf-8"))
    season = int(metadata["season"])
    week = int(metadata["week"])
    command = [
        str(UV),
        "run",
        "--no-sync",
        "nfl-ats",
        "weekly-run",
        "--season",
        str(season),
        "--week",
        str(week),
        "--refresh-player-data",
        "--skip-ingest",
        "--skip-prospective",
        "--skip-drift",
    ]
    if args.dry_run:
        return subprocess.run([*command, "--dry-run"], cwd=REPO, check=False).returncode
    subprocess.run(
        [sys.executable, str(REPO / "scripts" / "build_week_lineups.py")],
        cwd=REPO,
        check=True,
    )
    completed = subprocess.run(command, cwd=REPO, check=False)
    if completed.returncode != 0:
        return completed.returncode
    subprocess.run(
        [sys.executable, str(REPO / "scripts" / "build_week_lineups.py")],
        cwd=REPO,
        check=True,
    )
    subprocess.run(
        [str(UV), "run", "--no-sync", "nfl-ats", "publish-predictions", "--with-board"],
        cwd=REPO,
        check=True,
    )
    subprocess.run(
        [str(UV), "run", "--no-sync", "nfl-ats", "card-ledger-check"],
        cwd=REPO,
        check=False,
    )
    subprocess.run(
        [
            str(UV),
            "run",
            "--no-sync",
            "nfl-ats",
            "refresh-picks",
            "--publish-card",
            "--note",
            "lineups_refresh",
        ],
        cwd=REPO,
        check=False,
    )
    subprocess.run(
        [str(UV), "run", "--no-sync", "nfl-ats", "publish-board"],
        cwd=REPO,
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
