"""Refresh current depth charts and regenerate the active weekly forecast."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from nfl_ats.active_model import load_active_ats_model
from nfl_ats.public_board import active_artifact_path

REPO = Path(__file__).resolve().parents[1]
UV = REPO / ".tools" / "uv.exe"


def main() -> int:
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
    subprocess.run(
        [sys.executable, str(REPO / "scripts" / "build_week_lineups.py")],
        cwd=REPO,
        check=True,
    )
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
    completed = subprocess.run(command, cwd=REPO, check=False)
    if completed.returncode != 0:
        return completed.returncode
    # Rebuild after weekly-run so the public artifact is linked to the
    # forecast scored from the refreshed depth snapshot.
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
